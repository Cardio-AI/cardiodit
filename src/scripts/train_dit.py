"""
Training script for DiT4D (4D Diffusion Transformer) in the latent space of the
trained spatiotemporal VQ-GAN.
"""

import argparse
import math
import os
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import LambdaLR
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from omegaconf import OmegaConf

    from src.models.dit import DiT4D
    from src.models.ddpmscheduler import DDPMScheduler
    from src.models.flow_matching_scheduler import FlowMatchingScheduler
    from src.training.dit_trainer import DiTTrainer
    from src.data.dataloading import get_dit_dataloader
    from src.utils.wandb_utils import init_wandb, load_resume_run_id
    from src.utils.checkpointing import (
        CheckpointError,
        build_provenance,
        canonical_hash,
        load_latest_checkpoint,
        prepare_model_state_for_load,
        restore_rng_state,
    )
    from src.config_validation import validate_dit_config
except ModuleNotFoundError:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        np = torch = optim = LambdaLR = dist = DDP = OmegaConf = None
        DiT4D = DDPMScheduler = FlowMatchingScheduler = DiTTrainer = None
        get_dit_dataloader = init_wandb = load_resume_run_id = None
    else:
        raise


def build_scheduler(config):
    scheduler_type = config.get("scheduler_type", "ddpm")
    if scheduler_type == "flow_matching":
        return FlowMatchingScheduler(**config.scheduler)
    return DDPMScheduler(**config.scheduler)


def wrap_distributed_model(model, world_size, local_rank):
    """Apply the production DDP options without assuming distributed startup."""
    if world_size <= 1:
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        # ``static_graph=True`` is incompatible with the ``no_sync``
        # gradient-accumulation path used by DiTTrainer on the cluster's
        # PyTorch build (the reducer asserts in the second backward pass).
        static_graph=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--training_ids", type=str, required=True)
    parser.add_argument("--validation_ids", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage1_ckpt", type=str, default=None,
        help="Stage-1 checkpoint used to create the precomputed latents (hashed as provenance).",
    )
    parser.add_argument(
        "--stage1_cfg", type=str, default=None,
        help="Stage-1 config used to create the precomputed latents (hashed as provenance).",
    )
    parser.add_argument(
        "--allow_checkpoint_mismatch", action="store_true",
        help="Explicitly accept config/data/Stage-1 identity mismatches and record them.",
    )
    parser.add_argument(
        "--allow_legacy_scaling", action="store_true",
        help="Resume a proven legacy checkpoint with normalization plus non-unit scaling.",
    )
    parser.add_argument(
        "--allow_legacy_epoch_schedule", action="store_true",
        help="Explicitly convert legacy epoch schedule fields to update counts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fail unsafe/inconsistent configurations before touching CUDA or creating
    # a distributed process group.  The returned estimate is also checkpoint
    # provenance, so validation and recorded metadata cannot diverge.
    config = OmegaConf.load(args.config)
    preflight_checkpoint = None
    if args.allow_legacy_scaling:
        config_stem = Path(args.config).stem
        preflight_run = Path(args.output_dir) / (args.run_name or f"dit-{config_stem}")
        preflight_result = load_latest_checkpoint(preflight_run)
        preflight_checkpoint = (
            preflight_result[0] if preflight_result is not None else None
        )
    attention_estimate = validate_dit_config(
        config,
        legacy_checkpoint=preflight_checkpoint,
        allow_legacy_scaling=args.allow_legacy_scaling,
    )

    # -----------------------
    # DDP setup
    # -----------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        is_main = rank == 0
    else:
        rank = 0
        is_main = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Reproducibility
    # -----------------------
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # -----------------------
    # Config + run directory
    # -----------------------
    config_stem = Path(args.config).stem
    # Deterministic run dir per config so resume finds the same directory.
    # Pass --run_name to start a parallel run with the same config.
    run_name = args.run_name or f"dit-{config_stem}"
    group = f"dit-{config_stem}"

    run_dir = Path(args.output_dir) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    resolved_config = OmegaConf.to_container(config, resolve=True)
    latent_contract = dict(config.get("latent_preprocessing", {}))
    latent_contract.update({
        "use_precomputed_latents": True,
        "normalize_latents": bool(config.training.get("normalize_latents", False)),
        "scale_factor": float(config.training.get("scale_factor", 1.0)),
        "alpha_pool": str(config.training.get("alpha_pool", "linear")),
    })
    stage1_ckpt = args.stage1_ckpt or config.training.get("stage1_ckpt", None)
    stage1_cfg = args.stage1_cfg or config.training.get("stage1_cfg", None)
    if is_main:
        provenance = build_provenance(
            resolved_config=resolved_config,
            train_manifest=args.training_ids,
            validation_manifest=args.validation_ids,
            stage1_checkpoint=stage1_ckpt,
            stage1_config=stage1_cfg,
            latent_preprocessing=latent_contract,
            attention_estimate=attention_estimate,
            run_dir=run_dir,
        )
    else:
        provenance = None
    if world_size > 1:
        provenance_payload = [provenance]
        dist.broadcast_object_list(provenance_payload, src=0)
        provenance = provenance_payload[0]

    resume_result = load_latest_checkpoint(
        run_dir,
        expected_provenance=provenance,
        allow_mismatch=args.allow_checkpoint_mismatch,
    )
    dit_checkpoint = resume_result[0] if resume_result is not None else None
    checkpoint_path = resume_result[1] if resume_result is not None else None
    resume_overrides = resume_result[2] if resume_result is not None else []
    if resume_overrides:
        resume_overrides = [
            {**mismatch, "override": "--allow_checkpoint_mismatch"}
            for mismatch in resume_overrides
        ]

    # Re-use W&B history only when model state also resumes. If a run failed
    # before its first checkpoint, training restarts from update zero and must
    # receive a fresh tracking ID as well.
    wandb_id_file = run_dir / "wandb_id.txt"
    wandb_id = (
        load_resume_run_id(
            wandb_id_file,
            has_checkpoint=checkpoint_path is not None,
        )
        if is_main
        else None
    )
    wandb_run = None
    wandb_init_error = None
    wandb_init_exception = None
    if is_main:
        try:
            wandb_run = init_wandb(
                config=config,
                run_name=run_name,
                job_type="dit_train",
                tags=["stage2", "dit"],
                group=group,
                run_id=wandb_id,
            )
        except Exception as exc:
            wandb_init_exception = exc
            wandb_init_error = f"{type(exc).__name__}: {exc}"

    # All ranks must observe rank zero's W&B outcome before any rank enters
    # DDP construction. Otherwise rank zero can exit while peers hang in a
    # model-verification collective and report a misleading model mismatch.
    if world_size > 1:
        wandb_status = [wandb_init_error]
        dist.broadcast_object_list(wandb_status, src=0)
        wandb_init_error = wandb_status[0]
    if wandb_init_error is not None:
        error = RuntimeError(f"W&B initialization failed on rank 0: {wandb_init_error}")
        if wandb_init_exception is not None:
            raise error from wandb_init_exception
        raise error

    if is_main and wandb_run is not None and wandb_run.id != wandb_id:
        wandb_id_file.write_text(wandb_run.id)

    # -----------------------
    # Data
    # -----------------------
    patch_size = list(config.model.params.get("patch_size", [1, 1, 1, 1]))
    latent_time_pad_multiple = config.training.get("latent_time_pad_multiple", None)
    if latent_time_pad_multiple is None and len(patch_size) >= 4:
        latent_time_pad_multiple = int(patch_size[-1])
    if latent_time_pad_multiple is not None and int(latent_time_pad_multiple) <= 1:
        latent_time_pad_multiple = None
    if is_main and latent_time_pad_multiple is not None:
        print(
            "Latent temporal padding enabled: "
            f"T will be cyclically padded to a multiple of {latent_time_pad_multiple}; "
            "padded frames are masked from the loss."
        )

    train_loader, val_loader = get_dit_dataloader(
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        use_precomputed_latents=True,
        preload_latents=bool(config.training.get("preload_latents", True)),
        phase_dir=config.training.get("phase_dir", None),
        use_bucket_sampler=bool(config.training.get("use_bucket_sampler", False)),
        alpha_pool=config.training.get("alpha_pool", "linear"),
        batch_strategy=config.training.get("batch_strategy", None),
        variable_shape=bool(config.model.params.get("variable_shape", False)),
        latent_time_pad_multiple=latent_time_pad_multiple,
        allow_legacy_latents=bool(
            config.training.get("allow_legacy_latents", False)
        ),
    )
    padding_stats = getattr(train_loader.batch_sampler, "padding_stats", None)
    if is_main and callable(padding_stats):
        stats = padding_stats()
        print(
            "Training bucket padding: "
            f"{stats['padding_examples']}/{stats['padded_examples']} examples "
            f"({100.0 * stats['padding_fraction']:.2f}%)."
        )

    # -----------------------
    # DiT model
    # -----------------------
    model = DiT4D(**config.model.params).to(device)
    scheduler = build_scheduler(config)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"DiT4D parameters: {n_params:.1f}M")

    model = wrap_distributed_model(model, world_size, local_rank)

    # -----------------------
    # Optimizer + LR scheduler
    # -----------------------
    weight_decay = float(config.optim.get("weight_decay", 1e-4))
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        # Match parameter-name leaves only (e.g. ".bias", ".weight" of a norm
        # layer) to avoid catching unrelated modules whose names contain the
        # substring "norm" (e.g. "normalize_*").
        leaf = name.rsplit(".", 1)[-1]
        parent = name.rsplit(".", 2)[-2] if "." in name else ""
        is_no_decay = (
            leaf == "bias"
            or "norm" in parent.lower()
            or "layernorm" in parent.lower()
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.optim.lr,
        fused=(device.type == "cuda"),
    )

    min_lr_ratio = float(config.optim.get("min_lr_ratio", 0.01))
    grad_accum_steps = max(1, int(config.training.get("grad_accum_steps", 1)))
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    has_update_schedule = (
        config.optim.get("warmup_updates", None) is not None
        and config.optim.get("total_updates", None) is not None
    )
    if has_update_schedule:
        warmup_updates = int(config.optim.warmup_updates)
        total_updates = int(config.optim.total_updates)
        schedule_source = "explicit_update_counts"
    elif not args.allow_legacy_epoch_schedule:
        raise ValueError(
            "New runs require optim.warmup_updates and optim.total_updates so "
            "schedules are invariant to loader size and accumulation. Use "
            "--allow_legacy_epoch_schedule only for an explicit legacy conversion."
        )
    else:
        warmup_epochs = int(config.optim.get("warmup_epochs", 100))
        warmup_updates = warmup_epochs * updates_per_epoch
        total_updates = int(config.training.n_epochs) * updates_per_epoch
        schedule_source = "legacy_epoch_conversion"
        resume_overrides.append({
            "migration": "epoch_schedule_to_update_counts",
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "override": "--allow_legacy_epoch_schedule",
        })
    if warmup_updates < 0 or total_updates <= 0 or warmup_updates >= total_updates:
        raise ValueError("Update schedule requires 0 <= warmup_updates < total_updates")
    schedule_contract = {
        "unit": "optimizer_update",
        "kind": "warmup_cosine",
        "warmup_updates": warmup_updates,
        "total_updates": total_updates,
        "min_lr_ratio": min_lr_ratio,
        "source": schedule_source,
    }
    saved_schedule = (dit_checkpoint or {}).get("schedule_contract")
    if saved_schedule is not None and canonical_hash(saved_schedule) != canonical_hash(schedule_contract):
        if not args.allow_checkpoint_mismatch:
            raise CheckpointError(
                "Checkpoint optimizer-update schedule contract does not match the current run."
            )
        resume_overrides.append({
            "field": "schedule_contract",
            "checkpoint": saved_schedule,
            "current": schedule_contract,
            "override": "--allow_checkpoint_mismatch",
        })

    def _warmup_cosine(update):
        warmup = max(warmup_updates, 1)
        if update < warmup:
            return (update + 1) / warmup
        progress = (update - warmup) / max(total_updates - warmup - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr_scheduler = LambdaLR(optimizer, lr_lambda=_warmup_cosine)

    # -----------------------
    # Resume checkpoint
    # -----------------------
    start_epoch = 0
    best_loss = float("inf")

    if dit_checkpoint is not None:
        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")

        unavailable = [
            field for field in ("model", "optimizer", "epoch", "progress")
            if dit_checkpoint.get(field) is None
        ]
        if unavailable:
            raise CheckpointError(
                "Checkpoint is valid for limited inference but cannot resume training; "
                f"unavailable fields: {unavailable}"
            )

        raw_model = model.module if isinstance(model, DDP) else model
        raw_state, legacy_migrations = prepare_model_state_for_load(
            dit_checkpoint,
            set(raw_model.state_dict()),
            allow_legacy_rope4d_pos_embed=(
                str(config.model.params.get("pos_embed_mode", "sincos")).lower()
                == "rope4d"
            ),
        )
        if legacy_migrations:
            resume_overrides.extend(legacy_migrations)
            if is_main:
                print(f"Applied legacy checkpoint migration: {legacy_migrations[0]}")
        if isinstance(model, DDP):
            model.module.load_state_dict(raw_state)
        else:
            model.load_state_dict(raw_state)

        optimizer.load_state_dict(dit_checkpoint["optimizer"])
        # Optimizer state tensors land on CPU because of map_location="cpu".
        # Move them to the param device, otherwise optimizer.step mixes GPU
        # params with CPU momentum buffers → NaN / runtime error.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if dit_checkpoint.get("lr_scheduler") is not None:
            if dit_checkpoint.get("schedule_unit") == "optimizer_update":
                lr_scheduler.load_state_dict(dit_checkpoint["lr_scheduler"])
            elif not args.allow_checkpoint_mismatch:
                raise CheckpointError(
                    "Checkpoint LR schedule was not defined on optimizer updates. "
                    "Exact continuation under the corrected schedule requires "
                    "--allow_checkpoint_mismatch and will be recorded."
                )
            else:
                completed_updates = int(
                    dit_checkpoint["progress"]["optimizer_updates"]
                )
                lr_scheduler.last_epoch = completed_updates
                lr_scheduler._step_count = completed_updates + 1
                resume_overrides.append({
                    "migration": "lr_schedule_to_optimizer_updates",
                    "completed_updates": completed_updates,
                    "override": "--allow_checkpoint_mismatch",
                })

        progress = dit_checkpoint.get("progress") or {}
        start_epoch = (
            int(progress.get("epoch", dit_checkpoint["epoch"]))
            if "batch_in_epoch" in progress
            else dit_checkpoint["epoch"] + 1
        )
        best_loss = dit_checkpoint.get("best_loss", float("inf"))

        if world_size > 1:
            dist.barrier()

    # -----------------------
    # Trainer
    # -----------------------
    trainer = DiTTrainer(
        model=model,
        stage1=None,      # latents are precomputed
        scheduler=scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        is_main=is_main,
        start_epoch=start_epoch,
        best_loss=best_loss,
        wandb_run=wandb_run,
        checkpoint=dit_checkpoint,
        resolved_config=resolved_config,
        provenance=provenance,
        run_id=(wandb_run.id if wandb_run is not None else run_name),
        resume_overrides=resume_overrides,
        schedule_contract=schedule_contract,
    )

    if dit_checkpoint is not None and dit_checkpoint.get("ema") is not None:
        if is_main:
            print("Restoring EMA state")
        trainer.load_ema_state(dit_checkpoint["ema"])

    # Restore AMP GradScaler state so resume doesn't restart at the default
    # init_scale (65536) — that causes overflow on partially-trained models,
    # leading to repeated skipped steps and NaN params. Skipped under bf16
    # (no scaler) or when ckpt was written in bf16 mode.
    if (
        dit_checkpoint is not None
        and dit_checkpoint.get("scaler") is not None
        and trainer.scaler is not None
    ):
        if is_main:
            print("Restoring GradScaler state")
        trainer.scaler.load_state_dict(dit_checkpoint["scaler"])

    # Restore rank-local RNG only after all constructors and state loading have
    # completed, so the next training draw is exactly the saved next draw.
    if dit_checkpoint is not None and dit_checkpoint.get("rng") is not None:
        rng = dit_checkpoint["rng"]
        if isinstance(rng, dict) and "by_rank" in rng:
            restore_rng_state(rng, rank=rank)
        else:
            # Strict legacy compatibility: consume only fields that actually
            # exist in the legacy checkpoint.
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])

    # -----------------------
    # Train
    # -----------------------
    trainer.train()

    # -----------------------
    # Cleanup
    # -----------------------
    if is_main:
        if wandb_run:
            wandb_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

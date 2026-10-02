"""
Training script for the spatiotemporal VQ-GAN (Stage 1).

The VQ-GAN is a 3D model trained on individual 2D+t CMR slices (H, W, T).
After training, slices from a full 3D+t volume are encoded sequentially along
the depth (D) axis and stacked to form the 4D latent representation used by
the DiT (Stage 2).
"""

import argparse
import os
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.models.vqvae import VQVAE
from src.models.patchgan_discriminator import PatchDiscriminator
from src.losses.vqgan_loss import VQGANLoss
from src.training.vqgan_trainer import VQGANTrainer
from src.data.dataloading import get_vqgan_dataloader
from src.utils.wandb_utils import init_wandb, make_run_name
from src.utils.checkpointing import (
    CheckpointError,
    build_provenance,
    canonical_hash,
    load_latest_checkpoint,
    restore_rng_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--training_ids", type=str, required=True)
    parser.add_argument("--validation_ids", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_schedule_migration",
        action="store_true",
        help="Explicitly migrate an epoch-scheduled legacy checkpoint to update scheduling.",
    )
    parser.add_argument(
        "--allow_checkpoint_mismatch",
        action="store_true",
        help="Explicitly accept and record config/data provenance mismatches.",
    )
    return parser.parse_args()


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def resolve_cache_dir(cli_cache_dir, config):
    return cli_cache_dir or config.training.get("cache_dir", "/tmp/vqgan_cache")


def validate_vqgan_config(config, world_size: int) -> None:
    """Fail before process-group/model setup for unsafe Stage-1 settings."""
    model_params = config.get("model", {}).get("params", {})
    if int(world_size) > 1 and not bool(model_params.get("ddp_sync", False)):
        raise ValueError(
            "Distributed Stage-1 training with an EMA codebook requires "
            "model.params.ddp_sync=true. Unsynchronized EMA codebooks diverge "
            "across ranks."
        )
    dead_code_threshold = float(model_params.get("dead_code_threshold", 0.5))
    if dead_code_threshold < 0:
        raise ValueError("model.params.dead_code_threshold must be non-negative")


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config = OmegaConf.load(args.config)
    validate_vqgan_config(config, world_size)

    torch.set_float32_matmul_precision("high")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        is_main = rank == 0
    else:
        rank = 0
        is_main = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cache_dir = resolve_cache_dir(args.cache_dir, config)

    config_stem = Path(args.config).stem
    run_name = args.run_name or make_run_name("vqgan", args.config)
    group = f"vqgan-{config_stem}"

    run_dir = Path(args.output_dir) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    resolved_config = OmegaConf.to_container(config, resolve=True)
    preprocessing_contract = {
        "roi_size": list(config.training.roi_size),
        "target_frames": config.training.get("target_frames", None),
        "time_pad_multiple": config.training.get("time_pad_multiple", None),
        "spatial_permute": config.training.get("spatial_permute", None),
        "quantized": True,
    }
    if is_main:
        provenance = build_provenance(
            resolved_config=resolved_config,
            train_manifest=args.training_ids,
            validation_manifest=args.validation_ids,
            latent_preprocessing=preprocessing_contract,
            run_dir=run_dir,
        )
    else:
        provenance = None
    if world_size > 1:
        payload = [provenance]
        dist.broadcast_object_list(payload, src=0)
        provenance = payload[0]

    resume_result = load_latest_checkpoint(
        run_dir,
        expected_provenance=provenance,
        allow_mismatch=args.allow_checkpoint_mismatch,
    )

    writer_train = SummaryWriter(run_dir / "logs" / "train") if is_main else None
    writer_val = SummaryWriter(run_dir / "logs" / "val") if is_main else None

    wandb_run = init_wandb(
        config=config,
        run_name=run_name,
        job_type="vqgan_train",
        tags=["stage1", "vqgan"],
        group=group,
    ) if is_main else None

    # -----------------------
    # Data
    # -----------------------
    roi_size = tuple(config.training.roi_size)   # e.g. (224, 224, 32) = (H, W, T)
    target_frames = config.training.get("target_frames", roi_size[-1])
    time_pad_multiple = config.training.get("time_pad_multiple", None)

    spatial_permute = config.training.get("spatial_permute", None)
    if spatial_permute is not None:
        spatial_permute = tuple(spatial_permute)

    train_loader, val_loader = get_vqgan_dataloader(
        cache_dir=cache_dir,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        roi_size=roi_size,
        target_frames=target_frames,
        time_pad_multiple=time_pad_multiple,
        use_persistent=config.training.use_persistent,
        spatial_permute=spatial_permute,
    )

    # -----------------------
    # Models
    # -----------------------
    model = VQVAE(**config.model.params).to(device)
    discriminator = PatchDiscriminator(**config.discriminator.params).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

    # -----------------------
    # Loss
    # -----------------------
    loss_fn = VQGANLoss(
        perceptual_weight=config.losses.perceptual_weight,
        jukebox_weight=config.losses.get("jukebox_weight", 1.0),
        **config.losses.get("params", {}),
    ).to(device)

    # -----------------------
    # Optimizers
    # -----------------------
    optimizer_g = optim.Adam(model.parameters(), lr=config.training.base_lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.training.disc_lr, betas=(0.5, 0.9))
    # Preserve the configured epoch-scale decay while stepping the scheduler at
    # the actual optimizer-update cadence.
    if config.training.get("lr_update_gamma", None) is not None:
        update_gamma = float(config.training.lr_update_gamma)
        schedule_source = "explicit_update_gamma"
    elif not args.allow_schedule_migration:
        raise ValueError(
            "New Stage-1 runs require training.lr_update_gamma. The historical "
            "epoch lr_gamma conversion is available only with the explicit "
            "--allow_schedule_migration legacy flag."
        )
    else:
        epoch_gamma = float(config.training.get("lr_gamma", 0.999))
        update_gamma = epoch_gamma ** (1.0 / max(len(train_loader), 1))
        schedule_source = "legacy_epoch_gamma_conversion"
    if not 0.0 < update_gamma <= 1.0:
        raise ValueError("training.lr_update_gamma must be in (0, 1]")
    schedule_contract = {
        "unit": "optimizer_update",
        "kind": "exponential",
        "update_gamma": update_gamma,
        "source": schedule_source,
    }
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=update_gamma)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=update_gamma)

    # -----------------------
    # Resume checkpoint
    # -----------------------
    start_epoch = 0
    best_loss = float("inf")
    ckpt = resume_result[0] if resume_result is not None else None
    checkpoint_path = resume_result[1] if resume_result is not None else None
    resume_overrides = resume_result[2] if resume_result is not None else []

    if ckpt is not None:
        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")
        unavailable = [
            field
            for field in ("model", "discriminator", "optimizer_g", "optimizer_d", "epoch")
            if ckpt.get(field) is None
        ]
        if unavailable:
            raise CheckpointError(
                f"VQGAN checkpoint cannot resume; unavailable fields: {unavailable}"
            )
        raw_model = model.module if isinstance(model, DDP) else model
        raw_disc = discriminator.module if isinstance(discriminator, DDP) else discriminator
        raw_model.load_state_dict(ckpt["model"])
        raw_disc.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        _move_optimizer_state_to_device(optimizer_g, device)
        _move_optimizer_state_to_device(optimizer_d, device)
        # Override LRs from config (checkpoint state dict overwrites them otherwise)
        for pg in optimizer_g.param_groups:
            pg["lr"] = config.training.base_lr
        for pg in optimizer_d.param_groups:
            pg["lr"] = config.training.disc_lr
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("best_loss", float("inf"))
        progress = ckpt.get("progress")
        if ckpt.get("schedule_unit") == "optimizer_update":
            if ckpt.get("scheduler_g") is not None:
                scheduler_g.load_state_dict(ckpt["scheduler_g"])
            if ckpt.get("scheduler_d") is not None:
                scheduler_d.load_state_dict(ckpt["scheduler_d"])
        elif not args.allow_schedule_migration:
            raise RuntimeError(
                "Legacy VQGAN checkpoint schedules were epoch-based and have no "
                "optimizer-update contract. Re-run with --allow_schedule_migration "
                "only after accepting the recorded schedule migration."
            )
        else:
            completed_g = int(
                (progress or {}).get(
                    "optimizer_updates", start_epoch * len(train_loader)
                )
            )
            active_d_epochs = sum(
                1
                for epoch in range(start_epoch)
                if (
                    config.losses.adv_weight > 0
                    and (
                        epoch >= config.losses.adv_warmup
                        or epoch > 0
                    )
                )
            )
            completed_d = int(
                (progress or {}).get(
                    "discriminator_updates", active_d_epochs * len(train_loader)
                )
            )
            scheduler_g.last_epoch = completed_g
            scheduler_g._step_count = completed_g + 1
            scheduler_d.last_epoch = completed_d
            scheduler_d._step_count = completed_d + 1
            ckpt["progress"] = {
                **(progress or {}),
                "optimizer_updates": completed_g,
                "discriminator_updates": completed_d,
                "global_step": completed_g,
            }
            resume_overrides.append({
                "migration": "vqgan_lr_schedule_to_optimizer_updates",
                "generator_updates": completed_g,
                "discriminator_updates": completed_d,
                "override": "--allow_schedule_migration",
            })

        if world_size > 1:
            dist.barrier()

    saved_schedule = (ckpt or {}).get("schedule_contract")
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

    # -----------------------
    # Trainer
    # -----------------------
    trainer = VQGANTrainer(
        model=model,
        discriminator=discriminator,
        loss_fn=loss_fn,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        writer_train=writer_train,
        writer_val=writer_val,
        is_main=is_main,
        start_epoch=start_epoch,
        best_loss=best_loss,
        wandb_run=wandb_run,
        checkpoint=ckpt,
        resume_overrides=resume_overrides,
        resolved_config=resolved_config,
        provenance=provenance,
        run_id=run_name,
        schedule_contract=schedule_contract,
    )

    if ckpt is not None:
        if ckpt.get("scaler_g") is not None:
            trainer.scaler_g.load_state_dict(ckpt["scaler_g"])
        if ckpt.get("scaler_d") is not None:
            trainer.scaler_d.load_state_dict(ckpt["scaler_d"])

        rng = ckpt.get("rng")
        if rng is not None:
            if "by_rank" in rng:
                restore_rng_state(rng, rank=rank)
            else:
                # Explicit legacy adapter: old Stage-1 checkpoints captured
                # rank-zero/global CUDA state only and are not exact DDP resumes.
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                np.random.set_state(rng["numpy"])
                random.setstate(rng["python"])

    trainer.train()

    if is_main:
        writer_train.close()
        writer_val.close()
        if wandb_run:
            wandb_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

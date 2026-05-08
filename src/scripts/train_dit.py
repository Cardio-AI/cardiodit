"""
Train CardioDiT in the precomputed 4D VQ-GAN latent space.
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from src.data.dataloading import get_dit_dataloader
from src.models.ddpmscheduler import DDPMScheduler
from src.models.dit import DiT4D
from src.training.dit_trainer import DiTTrainer
from src.utils.wandb_utils import init_wandb, make_run_name


def normalize_config(config):
    """Accept both the submission-style ``dit:`` config and the repo config."""
    if "model" in config:
        return config

    if "dit" not in config:
        raise ValueError("Config must contain either a 'model' block or a 'dit' block.")

    defaults = {
        "model": {"params": config.dit.params},
        "scheduler": config.dit.scheduler,
        "training": {
            "n_epochs": int(config.dit.get("n_epochs", 3000)),
            "eval_freq": int(config.dit.get("eval_freq", 10)),
            "batch_size": int(config.dit.get("batch_size", 1)),
            "num_workers": int(config.dit.get("num_workers", 8)),
            "scale_factor": float(config.dit.get("scale_factor", 1.0)),
            "preload_latents": bool(config.dit.get("preload_latents", True)),
            "use_ema": bool(config.dit.get("use_ema", True)),
            "ema_decay": float(config.dit.get("ema_decay", 0.9999)),
            "grad_accum_steps": int(config.dit.get("grad_accum_steps", 1)),
        },
        "optim": {
            "lr": float(config.dit.get("base_lr", 1.0e-4)),
            "weight_decay": float(config.dit.get("weight_decay", 1.0e-4)),
            "warmup_epochs": int(config.dit.get("warmup_epochs", 100)),
            "min_lr_ratio": float(config.dit.get("min_lr_ratio", 0.01)),
        },
        "wandb": {
            "entity": config.get("wandb", {}).get("entity", None),
            "project": config.get("wandb", {}).get("project", "cardiodit"),
            "offline": config.get("wandb", {}).get("offline", False),
        },
    }
    return OmegaConf.create(defaults)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--training_ids", type=str, required=True)
    parser.add_argument("--validation_ids", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

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

    config = normalize_config(OmegaConf.load(args.config))
    run_name = args.run_name or make_run_name("dit", args.config)
    run_dir = Path(args.output_dir) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    wandb_run = (
        init_wandb(
            config=config,
            run_name=run_name,
            job_type="dit_train",
            tags=["stage2", "dit4d"],
            group=f"dit-{Path(args.config).stem}",
        )
        if is_main
        else None
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
    )

    model = DiT4D(**config.model.params).to(device)
    scheduler = DDPMScheduler(**config.scheduler)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"DiT4D parameters: {n_params:.1f}M")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    weight_decay = float(config.optim.get("weight_decay", 1e-4))
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower() or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(config.optim.lr),
    )

    warmup_epochs = int(config.optim.get("warmup_epochs", 100))
    min_lr_ratio = float(config.optim.get("min_lr_ratio", 0.01))
    total_epochs = int(config.training.n_epochs)

    def warmup_cosine(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

    checkpoint_path = run_dir / "last_checkpoint.pth"
    start_epoch = 0
    best_loss = float("inf")
    dit_checkpoint = None

    if checkpoint_path.exists():
        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")
        dit_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        raw_state = dit_checkpoint["model"]
        if isinstance(model, DDP):
            model.module.load_state_dict(raw_state)
        else:
            model.load_state_dict(raw_state)

        optimizer.load_state_dict(dit_checkpoint["optimizer"])
        if dit_checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(dit_checkpoint["lr_scheduler"])

        start_epoch = int(dit_checkpoint["epoch"]) + 1
        best_loss = float(dit_checkpoint.get("best_loss", float("inf")))

        if world_size > 1:
            dist.barrier()

    trainer = DiTTrainer(
        model=model,
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
    )

    if dit_checkpoint is not None and dit_checkpoint.get("ema") is not None:
        trainer.load_ema_state(dit_checkpoint["ema"])

    trainer.train()

    if is_main and wandb_run is not None:
        wandb_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

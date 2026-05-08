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


def normalize_config(config):
    """Accept both the compact ``stage1:`` config and the repo config."""
    if "model" in config:
        return config

    if "stage1" not in config:
        raise ValueError("Config must contain either a 'model' block or a 'stage1' block.")

    perceptual_params = {}
    if "perceptual_network" in config:
        perceptual_params = config.perceptual_network.get("params", {})

    defaults = {
        "model": {"params": config.stage1.params},
        "discriminator": config.discriminator,
        "training": {
            "n_epochs": int(config.stage1.get("n_epochs", 500)),
            "eval_freq": int(config.stage1.get("eval_freq", 10)),
            "batch_size": int(config.stage1.get("batch_size", 4)),
            "num_workers": int(config.stage1.get("num_workers", 8)),
            "base_lr": float(config.stage1.get("base_lr", 5.0e-5)),
            "disc_lr": float(config.stage1.get("disc_lr", 2.0e-5)),
            "lr_gamma": float(config.stage1.get("lr_gamma", 0.9999)),
            "roi_size": list(config.stage1.get("roi_size", [256, 256, 32])),
            "target_frames": int(config.stage1.get("target_frames", 32)),
            "use_persistent": bool(config.stage1.get("use_persistent", True)),
        },
        "losses": {
            "perceptual_weight": float(config.stage1.get("perceptual_weight", 0.002)),
            "jukebox_weight": float(config.stage1.get("jukebox_weight", 0.0)),
            "adv_weight": float(config.stage1.get("adv_weight", 0.005)),
            "adv_warmup": int(config.stage1.get("adv_warmup", 50)),
            "params": {
                "perceptual_params": perceptual_params,
                "jukebox_params": {"spatial_dims": 3},
            },
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
    parser.add_argument("--cache_dir", type=str, required=False)
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

    config_stem = Path(args.config).stem
    run_name = args.run_name or make_run_name("vqgan", args.config)
    group = f"vqgan-{config_stem}"

    run_dir = Path(args.output_dir) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

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

    train_loader, val_loader = get_vqgan_dataloader(
        cache_dir=args.cache_dir or "/tmp/cardiodit_vqgan_cache",
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        roi_size=roi_size,
        target_frames=target_frames,
        use_persistent=config.training.use_persistent,
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
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=config.training.get("lr_gamma", 0.999))
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=config.training.get("lr_gamma", 0.999))

    # -----------------------
    # Resume checkpoint
    # -----------------------
    checkpoint_path = run_dir / "last_checkpoint.pth"
    start_epoch = 0
    best_loss = float("inf")

    if checkpoint_path.exists():
        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_disc = discriminator.module if isinstance(discriminator, DDP) else discriminator
        raw_model.load_state_dict(ckpt["model"])
        raw_disc.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        # Override LRs from config (checkpoint state dict overwrites them otherwise)
        for pg in optimizer_g.param_groups:
            pg["lr"] = config.training.base_lr
        for pg in optimizer_d.param_groups:
            pg["lr"] = config.training.disc_lr
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("best_loss", float("inf"))

        if world_size > 1:
            dist.barrier()

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
    )

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

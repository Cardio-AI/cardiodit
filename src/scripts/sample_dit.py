"""
Sample 4D CMR volumes from a trained CardioDiT checkpoint.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import amp

from src.models.ddimscheduler import DDIMScheduler
from src.models.ddpmscheduler import DDPMScheduler
from src.models.dit import DiT4D
from src.models.vqvae import VQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained CardioDiT")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_cfg", type=str, required=True)
    parser.add_argument("--diff_cfg", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument(
        "--latent_shape",
        type=int,
        nargs=5,
        required=True,
        metavar=("C", "D", "H", "W", "T"),
        help="Shape of one latent, for example: 8 10 28 28 8",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_stage1(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = VQVAE(**cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model_keys = set(model.state_dict().keys())
    filtered = {key: value for key, value in state_dict.items() if key in model_keys}
    model.load_state_dict(filtered, strict=False)
    return model.to(device).eval().requires_grad_(False)


def load_dit(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = DiT4D(**cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = dict(ckpt.get("model", ckpt))
    if ckpt.get("ema") is not None:
        state_dict.update(ckpt["ema"]["shadow"])
    model.load_state_dict(state_dict)
    return model.to(device).eval().requires_grad_(False), cfg


def build_scheduler(name, diff_cfg):
    cfg = dict(diff_cfg.scheduler)
    if name == "ddpm":
        return DDPMScheduler(**cfg)
    if name == "ddim":
        return DDIMScheduler(**cfg)
    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def decode_latent(stage1, latent_4d):
    """Decode (C, D, H, W, T) latent to (H, W, D, T) image volume."""
    depth = latent_4d.shape[1]
    slices = []
    for d_idx in range(depth):
        z = latent_4d[:, d_idx, :, :, :].unsqueeze(0)
        recon = stage1.decode_stage_2_outputs(z)
        slices.append(recon[0])
    volume = torch.stack(slices, dim=1)  # (1, D, H, W, T)
    return volume[0].permute(1, 2, 0, 3).contiguous()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)
    dit, diff_cfg = load_dit(args.diff_cfg, args.diff_ckpt, device)
    scheduler = build_scheduler(args.scheduler, diff_cfg)
    scheduler.set_timesteps(args.timesteps, device=device)

    channels, depth, height, width, time = args.latent_shape
    print(
        f"Sampling {args.n_samples} volumes with {args.scheduler.upper()} "
        f"({args.timesteps} steps)"
    )

    for sample_idx in range(args.n_samples):
        x = torch.randn(1, channels, depth, height, width, time, device=device)

        with amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for timestep in scheduler.timesteps:
                t_int = int(timestep.item())
                t_batch = torch.full((1,), t_int, device=device, dtype=torch.long)
                model_output = dit(x, t=t_batch, y=None)
                if args.scheduler == "ddim":
                    x, _ = scheduler.step(model_output, t_int, x, eta=args.eta)
                else:
                    x, _ = scheduler.step(model_output, t_int, x)

        x = x / args.scale_factor
        volume = decode_latent(stage1, x[0]).float().cpu().clamp(-1.0, 1.0).numpy()

        out_path = out_dir / f"sample_{sample_idx:03d}.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)), out_path)
        print(f"Saved {out_path} shape={list(volume.shape)}")

        del x, volume
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

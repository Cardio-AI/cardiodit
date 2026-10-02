"""
Compute the latent scale factor (1 / std(z)) from precomputed 4D latent files.

The scale factor is used as ``training.scale_factor`` in the DiT config to
normalise the latent distribution to unit variance before diffusion training.

Usage:
    python src/scripts/compute_scale_factor.py \\
        --latents_csv data/latents/train/latents.csv \\
        --limit 200
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.data.latent_contract import validate_latent_payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latents_csv", required=True,
        help="CSV with precomputed latent paths (column: image)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of latents to use (default: all)",
    )
    return parser.parse_args()


def streaming_latent_std(paths):
    total_sum = torch.tensor(0.0, dtype=torch.float64)
    total_sq_sum = torch.tensor(0.0, dtype=torch.float64)
    total_count = 0

    for p in paths:
        z, _ = validate_latent_payload(
            torch.load(p, map_location="cpu", weights_only=False), allow_legacy=True
        )
        flat = torch.as_tensor(z, dtype=torch.float64).reshape(-1)
        total_sum += flat.sum()
        total_sq_sum += flat.square().sum()
        total_count += flat.numel()

    if total_count < 2:
        raise ValueError("Need at least two latent values to compute an unbiased std.")

    variance = (total_sq_sum - (total_sum * total_sum) / total_count) / (total_count - 1)
    return variance.clamp_min(0.0).sqrt().item()


def main():
    args = parse_args()

    df = pd.read_csv(args.latents_csv)
    manifest = Path(args.latents_csv).expanduser().resolve()
    paths = [str(Path(p) if Path(p).is_absolute() else manifest.parent / p) for p in df["image"].tolist()]

    if args.limit:
        paths = paths[: args.limit]

    print(f"Computing scale factor over {len(paths)} latents...")

    std = streaming_latent_std(paths)
    scale_factor = 1.0 / std

    print(f"\n  std(z)       = {std:.6f}")
    print(f"  scale_factor = {scale_factor:.6f}  (1 / std)")
    print(f"\nAdd to your diffusion config:")
    print(f"  training:")
    print(f"    scale_factor: {scale_factor:.6f}")


if __name__ == "__main__":
    main()

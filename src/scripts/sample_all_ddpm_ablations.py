"""
Sample 5 volumes from each DDPM ablation model (00-06) using full 1000-step DDPM.

For each ablation run in outputs/{00..06}_*/, loads last_checkpoint.pth with its
matching config from configs/transformer/test_configs/ddpm/, then calls
sample_dit.py with the per-config scale_factor.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()
OUTPUTS_DIR = RUNS_ROOT / "outputs"
DDPM_CFG_DIR = REPO_ROOT / "configs" / "transformer" / "test_configs" / "ddpm"
SAMPLE_SCRIPT = REPO_ROOT / "src" / "scripts" / "sample_dit.py"

ABLATIONS = [
    ("00_baseline",          "00_baseline.yaml"),
    ("01_dropout_zero",      "01_dropout_zero.yaml"),
    ("02_aniso_pos_embed",   "02_aniso_pos_embed.yaml"),
    ("03_zero_terminal_snr", "03_zero_terminal_snr.yaml"),
    ("04_offset_noise",      "04_offset_noise.yaml"),
    ("05_normalize_latents", "05_normalize_latents.yaml"),
    ("06_self_conditioning", "06_self_conditioning.yaml"),
]

LATENT_SHAPE = ["8", "10", "28", "28", "8"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--output_root", type=str, default=str(RUNS_ROOT / "samples"))
    p.add_argument("--spacing", type=float, nargs=4, default=[10.0, 1.7, 1.7, 1.0],
                   metavar=("D", "H", "W", "T"),
                   help="Image-space voxel spacing (D,H,W mm; T frames). "
                        "Same across ablations (shared data + stage1).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--weights", default="ema", choices=["raw", "ema", "both"])
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="Restrict to these run names (e.g. 00_baseline 03_zero_terminal_snr).")
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_cfg = None
    stage1_ckpt = None

    runs = ABLATIONS if args.only is None else [(r, c) for r, c in ABLATIONS if r in args.only]
    if not runs:
        print(f"No matching runs for --only={args.only}", file=sys.stderr)
        sys.exit(1)

    failures = []
    for run_name, cfg_name in runs:
        run_dir = OUTPUTS_DIR / run_name
        ckpt = run_dir / "last_checkpoint.pth"
        cfg_path = DDPM_CFG_DIR / cfg_name

        if not ckpt.is_file():
            print(f"[skip] {run_name}: no last_checkpoint.pth")
            continue
        if not cfg_path.is_file():
            print(f"[skip] {run_name}: missing config {cfg_path}")
            continue

        cfg = OmegaConf.load(cfg_path)
        scale_factor = float(cfg.training.scale_factor)
        s1_cfg = cfg.training.sample_log.stage1_cfg
        s1_ckpt = cfg.training.sample_log.stage1_ckpt
        if stage1_cfg is None:
            stage1_cfg, stage1_ckpt = s1_cfg, s1_ckpt
        elif (s1_cfg, s1_ckpt) != (stage1_cfg, stage1_ckpt):
            print(f"[warn] {run_name}: stage1 differs from first run, using its own.")

        sample_dir = output_root / run_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(SAMPLE_SCRIPT),
            "--stage1_cfg", str(s1_cfg),
            "--stage1_ckpt", str(s1_ckpt),
            "--diff_cfg", str(cfg_path),
            "--diff_ckpt", str(ckpt),
            "--output_dir", str(sample_dir),
            "--n_samples", str(args.n_samples),
            "--timesteps", str(args.timesteps),
            "--scheduler", "ddpm",
            "--scale_factor", str(scale_factor),
            "--latent_shape", *LATENT_SHAPE,
            "--spacing", *[str(s) for s in args.spacing],
            "--device", args.device,
            "--weights", args.weights,
            "--decoder_mode", "both",
        ]

        print(f"\n=== {run_name} | scale_factor={scale_factor} | "
              f"ckpt={ckpt.name} ===")
        print(" ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"[fail] {run_name}: sample_dit.py exited {result.returncode}",
                  file=sys.stderr)
            failures.append((run_name, result.returncode))

    if failures:
        return 1
    print("\nAll ablations done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

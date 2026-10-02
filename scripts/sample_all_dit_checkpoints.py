#!/usr/bin/env python
"""
Sample every matching DiT checkpoint under a run root.

This is a cluster-friendly wrapper around ``src/scripts/sample_dit.py``. It
prints every command in dry-run mode and never assumes site-specific paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()
sys.path.insert(0, str(REPO_ROOT))

from src.utils.sample_integrity import resolve_checkpoint_reference

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        OmegaConf = None
    else:
        raise


SAMPLE_SCRIPT = REPO_ROOT / "src" / "scripts" / "sample_dit.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample all DiT checkpoints in a run tree")
    parser.add_argument("--runs_root", required=True, help="Directory containing DiT run directories")
    parser.add_argument("--configs_root", required=True, help="Directory containing matching YAML configs")
    parser.add_argument("--stage1_cfg", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint_glob", default="last_checkpoint.pth")
    parser.add_argument("--config_glob", default="*.yaml")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--scheduler", default="ddim",
                        choices=["ddpm", "ddim", "dpm_pp", "flow_matching"])
    parser.add_argument("--weights", default="ema", choices=["ema", "raw", "both"])
    parser.add_argument("--decoder_mode", default="both", choices=["both", "direct", "quantized"])
    parser.add_argument("--latent_shape", type=int, nargs=5, default=None)
    parser.add_argument("--T_latent", type=int, default=None)
    parser.add_argument("--spacing", type=float, nargs=4, default=[10.0, 1.7, 1.7, 1.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def find_config(configs_root: Path, run_name: str, config_glob: str) -> Path | None:
    exact = configs_root / f"{run_name}.yaml"
    if exact.exists():
        return exact
    matches = sorted(configs_root.rglob(config_glob))
    for path in matches:
        if path.stem == run_name:
            return path
    for path in matches:
        if path.stem in run_name or run_name in path.stem:
            return path
    return None


def command_for(args, run_name: str, ckpt: Path, cfg_path: Path) -> list[str]:
    cfg = OmegaConf.load(cfg_path)
    scale_factor = float(cfg.training.get("scale_factor", 1.0))
    cmd = [
        sys.executable, str(SAMPLE_SCRIPT),
        "--stage1_cfg", args.stage1_cfg,
        "--stage1_ckpt", args.stage1_ckpt,
        "--diff_cfg", str(cfg_path),
        "--diff_ckpt", str(ckpt),
        "--output_dir", str(Path(args.output_root) / run_name / ckpt.stem / args.weights),
        "--n_samples", str(args.n_samples),
        "--timesteps", str(args.timesteps),
        "--scheduler", args.scheduler,
        "--scale_factor", str(scale_factor),
        "--weights", args.weights,
        "--decoder_mode", args.decoder_mode,
        "--spacing", *[str(v) for v in args.spacing],
        "--device", args.device,
    ]
    if args.latent_shape is not None:
        cmd.extend(["--latent_shape", *[str(v) for v in args.latent_shape]])
    if args.T_latent is not None:
        cmd.extend(["--T_latent", str(args.T_latent)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    configs_root = Path(args.configs_root)
    checkpoints = sorted(runs_root.rglob(args.checkpoint_glob))
    checkpoints = [resolve_checkpoint_reference(path) for path in checkpoints]
    if not checkpoints:
        checkpoints = [
            resolve_checkpoint_reference(pointer)
            for pointer in sorted(runs_root.rglob("last_checkpoint.json"))
        ]
    if not checkpoints:
        print(f"No checkpoints found under {runs_root} matching {args.checkpoint_glob}")
        return 1

    failures = []
    for ckpt in checkpoints:
        run_name = ckpt.parent.name
        cfg_path = find_config(configs_root, run_name, args.config_glob)
        if cfg_path is None:
            print(f"[skip] {run_name}: no matching config in {configs_root}")
            continue
        cmd = command_for(args, run_name, ckpt, cfg_path)
        print(" ".join(cmd))
        if not args.dry_run:
            status = subprocess.run(cmd, cwd=REPO_ROOT).returncode
            if status != 0:
                failures.append((run_name, status))

    if failures:
        for run_name, status in failures:
            print(f"[fail] {run_name}: exit {status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

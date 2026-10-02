"""
Skeleton evaluator for temporal descriptor quality.

The script reports descriptor monotonicity and adjacent-step smoothness. It is
kept deliberately data-format agnostic; image-level motion metrics can be added
once the motionfield/phi_t sidecar format is finalized.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from src.data.temporal_alignment import load_descriptor_sidecar, normalize_phase


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate temporal descriptor sidecars")
    parser.add_argument("sidecars", nargs="+")
    parser.add_argument("--motionfield", action="store_true",
                        help="Reserved for future phi_t sidecars; currently not implemented.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.motionfield:
        raise NotImplementedError(
            "motionfield evaluation is blocked until the phi_t sidecar format is finalized."
        )

    for path in args.sidecars:
        alpha = normalize_phase(load_descriptor_sidecar(path))
        diffs = alpha[1:] - alpha[:-1] if alpha.numel() > 1 else torch.zeros(1)
        monotone_fraction = float((diffs >= -1e-6).float().mean())
        mean_step = float(diffs.abs().mean())
        max_step = float(diffs.abs().max())
        print(
            f"{path}: length={alpha.numel()} "
            f"monotone_fraction={monotone_fraction:.3f} "
            f"mean_abs_step={mean_step:.4f} max_abs_step={max_step:.4f}"
        )


if __name__ == "__main__":
    main()

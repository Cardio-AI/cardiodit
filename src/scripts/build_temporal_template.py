"""
Build a DTW temporal descriptor template from alpha_t/phase sidecars.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from src.data.temporal_alignment import build_dtw_template, load_descriptor_sidecar


def parse_args():
    parser = argparse.ArgumentParser(description="Build a DTW template from temporal descriptors")
    parser.add_argument("sidecars", nargs="+", help="Descriptor sidecars (.pt/.npy/.npz/.json/.csv)")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--target_length", type=int, default=None)
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--window", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    descriptors = [load_descriptor_sidecar(path) for path in args.sidecars]
    template = build_dtw_template(
        descriptors,
        target_length=args.target_length,
        n_iters=args.n_iters,
        window=args.window,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"alpha_t": template}, output)
    print(f"Saved temporal template to {output} length={template.numel()}")


if __name__ == "__main__":
    main()

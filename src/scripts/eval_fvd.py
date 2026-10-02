"""
Compute Fréchet Video Distance (FVD) between generated samples and a
reference set of 4D CMR cines.

Status: skeleton.  Real-impl path requires an I3D backbone (StyleGAN-V weights
are the standard source — verify license before use).  Frames are taken along
the temporal axis with per-volume FVD; report mean over depth slices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description="FVD for generated cine cardiac MRI")
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--i3d_weights", required=True, help="Path to I3D backbone weights")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def compute_fvd(samples_dir: Path, reference_dir: Path, i3d_weights: Path, device: str) -> float:
    raise NotImplementedError(
        "FVD computation not implemented yet — confirm I3D weights source/"
        "licence before wiring this up."
    )


def main():
    args = parse_args()
    score = compute_fvd(
        Path(args.samples_dir), Path(args.reference_dir),
        i3d_weights=Path(args.i3d_weights), device=args.device,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"fvd": float(score)}, f, indent=2)
    print(f"FVD = {score:.4f}  →  {args.output_json}")


if __name__ == "__main__":
    main()

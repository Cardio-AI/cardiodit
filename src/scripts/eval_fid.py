"""
Compute Fréchet Inception Distance (FID) between generated samples and a
reference set of 4D CMR cines.

Status: skeleton.  Real-impl path will use torchmetrics-fidelity or the MONAI
FID implementation; volumes are projected to (T, 3, H, W) per-subject and
features are pooled across z-slices.  Must produce a numeric score on Exp 1
baseline samples before any other experiment is trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description="FID for generated cine cardiac MRI")
    parser.add_argument("--samples_dir", required=True, help="Directory of generated .nii.gz cines")
    parser.add_argument("--reference_dir", required=True, help="Directory of reference .nii.gz cines")
    parser.add_argument("--output_json", required=True, help="Where to write the FID score")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def compute_fid(samples_dir: Path, reference_dir: Path, device: str, batch_size: int) -> float:
    raise NotImplementedError(
        "FID computation not implemented yet — wire up torchmetrics-fidelity or "
        "MONAI's FID once Exp 1 baseline samples are available."
    )


def main():
    args = parse_args()
    score = compute_fid(
        Path(args.samples_dir), Path(args.reference_dir),
        device=args.device, batch_size=args.batch_size,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"fid": float(score)}, f, indent=2)
    print(f"FID = {score:.4f}  →  {args.output_json}")


if __name__ == "__main__":
    main()

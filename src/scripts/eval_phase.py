"""
Evaluate cardiac phase alignment of generated samples by running the Mueller
et al. keyframe detector and comparing detected ED/MS/ES/PF/MD positions
against expected normalised-cycle locations.

Status: skeleton.  Depends on the cmr-multi-view-phase-detection inference
script (https://github.com/Cardio-AI/cmr-multi-view-phase-detection.git).
Reports mean absolute frame error per keyframe and an aggregate metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


KEYFRAMES = ("ED", "MS", "ES", "PF", "MD")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-alignment evaluation")
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--reference_dir", required=True,
                        help="Reference cines used to derive expected normalised positions, "
                             "or a JSON of canonical targets.")
    parser.add_argument("--mueller_weights", required=True,
                        help="Phase-detector checkpoint from cmr-multi-view-phase-detection.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def compute_phase_errors(
    samples_dir: Path,
    reference_dir: Path,
    mueller_weights: Path,
    device: str,
) -> dict:
    raise NotImplementedError(
        "Phase eval not implemented yet — wire Mueller et al. detector once "
        "the inference script and weights are available locally."
    )


def main():
    args = parse_args()
    result = compute_phase_errors(
        Path(args.samples_dir), Path(args.reference_dir),
        mueller_weights=Path(args.mueller_weights), device=args.device,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

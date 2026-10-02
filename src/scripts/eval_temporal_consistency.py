"""
Measure temporal smoothness of generated cines via optical-flow magnitude
between consecutive frames.

Status: skeleton.  Real-impl uses torchvision's RAFT (or PWC-Net) per
(depth, time) slice and reports mean / std flow magnitude across samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description="Optical-flow temporal-consistency metric")
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow_backend", default="raft", choices=("raft", "pwcnet"))
    return parser.parse_args()


def compute_temporal_consistency(samples_dir: Path, device: str, flow_backend: str) -> dict:
    raise NotImplementedError(
        "Temporal-consistency eval not implemented yet — wire RAFT once Exp 1 "
        "baseline samples exist."
    )


def main():
    args = parse_args()
    result = compute_temporal_consistency(
        Path(args.samples_dir), device=args.device, flow_backend=args.flow_backend,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

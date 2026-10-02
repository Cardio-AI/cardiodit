"""
Prepare simple temporal descriptor sidecars from an image CSV.

This helper writes a uniformly sampled alpha_t descriptor per row. It is a
baseline for datasets without external cardiac phase estimates; replace these
sidecars with scanner/model-derived descriptors when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from src.data.temporal_alignment import linear_phase


def parse_args():
    parser = argparse.ArgumentParser(description="Write linear alpha_t sidecars for a CSV")
    parser.add_argument("--csv", required=True, help="Input CSV with image column")
    parser.add_argument("--output_dir", required=True, help="Descriptor output directory")
    parser.add_argument("--frames", type=int, required=True, help="Descriptor length")
    parser.add_argument("--output_csv", default=None, help="Optional CSV with alpha_t_path column")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _stem(path: str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    rows = []
    for _, row in df.iterrows():
        image = str(row["image"])
        out_path = output_dir / f"{_stem(image)}.pt"
        if args.force or not out_path.exists():
            torch.save({"alpha_t": linear_phase(args.frames)}, out_path)
            print(f"wrote {out_path}")
        rows.append({**row.to_dict(), "alpha_t_path": str(out_path)})

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

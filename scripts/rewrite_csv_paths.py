#!/usr/bin/env python
"""
Rewrite path prefixes in a CSV into a new output file.

The script is intentionally non-destructive: --output is required unless
--dry_run is used, and the input file is never modified in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite CSV path prefixes safely")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--column", default="image")
    parser.add_argument("--old_prefix", required=True)
    parser.add_argument("--new_prefix", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None and not args.dry_run:
        raise SystemExit("--output is required unless --dry_run is set")

    import pandas as pd

    df = pd.read_csv(args.input)
    if args.column not in df.columns:
        raise SystemExit(f"Missing column {args.column!r} in {args.input}")

    old = str(args.old_prefix)
    new = str(args.new_prefix)
    rewritten = df.copy()
    mask = rewritten[args.column].astype(str).str.startswith(old)
    rewritten.loc[mask, args.column] = (
        rewritten.loc[mask, args.column].astype(str).str.replace(old, new, n=1, regex=False)
    )
    print(f"rows={len(df)} rewritten={int(mask.sum())}")

    if args.dry_run:
        print(rewritten.head().to_string(index=False))
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rewritten.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

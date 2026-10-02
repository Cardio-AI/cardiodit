#!/usr/bin/env python
"""Migrate checksum-valid quantized samples into RUN/CHECKPOINT/sample_NNN.nii.gz."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()
sys.path.insert(0, str(REPO_ROOT))

from src.utils.checkpointing import atomic_json_save, sha256_file
from src.utils.sample_integrity import (
    atomic_file_save,
    completion_matches,
    flat_completion_matches,
    publish_flat_completion_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--flow_matching_timesteps", type=int, default=100)
    return parser.parse_args()


def is_target_run(name: str) -> bool:
    return re.match(r"^F\d\d", name) is not None or name.startswith("S1_")


def sampler_label(run_name: str, flow_steps: int) -> str:
    if "flow_matching" in run_name:
        return f"trained_flow_matching_{flow_steps}steps"
    if run_name.startswith("F00_"):
        return "trained_ddpm_300steps"
    return "trained_ddpm_1000steps"


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != sha256_file(source)
        ):
            raise RuntimeError(f"Existing destination differs from source: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        atomic_file_save(
            destination,
            lambda temporary: shutil.copyfile(source, temporary),
            suffix=".nii.gz",
        )
        return "copied"


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    output_root = Path(args.output_root).resolve()
    runs_root = RUNS_ROOT / "outputs" / "dit"
    counters = {"migrated": 0, "linked": 0, "copied": 0, "existing": 0, "skipped": 0}

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if not is_target_run(run_dir.name):
            continue
        label = sampler_label(run_dir.name, args.flow_matching_timesteps)
        checkpoints = []
        for checkpoint in run_dir.glob("checkpoint_update_*.pth"):
            match = re.fullmatch(r"checkpoint_update_(\d+)\.pth", checkpoint.name)
            if match and int(match.group(1)) % 50000 == 0:
                checkpoints.append(checkpoint)

        for checkpoint in sorted(checkpoints):
            flat_dir = output_root / run_dir.name / checkpoint.stem
            for sample_index in range(args.n_samples):
                old_dir = (
                    legacy_root
                    / run_dir.name
                    / checkpoint.stem
                    / "ema"
                    / label
                    / f"sample_{sample_index:03d}"
                )
                old_manifest_path = old_dir / "completion_manifest.json"
                if not old_manifest_path.is_file():
                    counters["skipped"] += 1
                    continue
                old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
                old_request = old_manifest.get("request", {}).get("payload")
                if not isinstance(old_request, dict) or not completion_matches(old_dir, old_request):
                    raise RuntimeError(f"Legacy completion is not checksum-valid: {old_dir}")
                if "quantized" not in old_request.get("decoder_modes", []):
                    counters["skipped"] += 1
                    continue

                request = dict(old_request)
                request["decoder_modes"] = ["quantized"]
                if flat_completion_matches(flat_dir, sample_index, request):
                    counters["existing"] += 1
                    continue

                source = old_dir / "decode_quantized" / f"sample_{sample_index:03d}.nii.gz"
                destination = flat_dir / f"sample_{sample_index:03d}.nii.gz"
                disposition = link_or_copy(source, destination)
                counters[disposition] += 1

                metadata = json.loads((old_dir / "sample_metadata.json").read_text(encoding="utf-8"))
                metadata["decoder_modes"] = ["quantized"]
                metadata["decoder_quantization"] = True
                metadata_path = flat_dir / ".metadata" / f"sample_{sample_index:03d}.json"
                atomic_json_save(metadata, metadata_path)
                publish_flat_completion_manifest(
                    flat_dir,
                    sample_index,
                    request,
                    [destination, metadata_path],
                )
                if not flat_completion_matches(flat_dir, sample_index, request):
                    raise RuntimeError(f"Migrated completion failed validation: {destination}")
                counters["migrated"] += 1

    print(" ".join(f"{key}={value}" for key, value in counters.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

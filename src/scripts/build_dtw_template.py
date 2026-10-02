"""
Build the Exp 5 descriptor-DTW temporal template.

Example:

python src/scripts/build_dtw_template.py \
    --csv train.csv \
    --sidecar_dir data \
    --keyframe_dir data/keyframes \
    --output data/dtw_template.pt \
    --max_warp_fraction 1.0

python src/scripts/build_dtw_template.py \
    --csv train.csv \
    --sidecar_dir data \
    --keyframe_dir data/keyframes \
    --metadata_csv data/dataset_information.csv \
    --group_by_label \
    --output_dir data/dtw_templates \
    --max_warp_fraction 1.0
"""

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import RUNS_ROOT

import pandas as pd
import torch

from src.utils.descriptor_io import (
    DESCRIPTOR_KEYS,
    find_descriptor_sidecar,
    load_1d_tensor,
)
from src.utils.temporal_align import build_dtw_template


def parse_args():
    parser = argparse.ArgumentParser(description="Build canonical DTW descriptor template")
    parser.add_argument("--csv", required=True, help="Training CSV used to define template subjects")
    parser.add_argument("--sidecar_dir", required=True,
                        help="Sidecar root containing descriptors as .pt or .npy under "
                             "alpha_t/, descriptors/, or motion_descriptor/, and "
                             "optionally keyframes/")
    parser.add_argument("--keyframe_dir", default=None,
                        help="Optional directory containing <subject_id>.json keyframes. "
                             "If omitted, sidecar_dir/keyframes/ and sidecar_dir/ are tried.")
    parser.add_argument("--output", default=None,
                        help="Output .pt path. Defaults to CARDIODIT_RUNS_DIR/temporal_alignment/dtw_template.pt")
    parser.add_argument("--target_frames", type=int, default=32)
    parser.add_argument("--n_iters", type=int, default=5,
                        help="Number of interval-wise DTW template refinement iterations")
    parser.add_argument("--aggregation", choices=("median", "mean"), default="median")
    parser.add_argument("--max_warp_fraction", type=float, default=0.5,
                        help="Normalised Sakoe-Chiba band used within each keyframe interval")
    parser.add_argument("--non_diagonal_penalty", type=float, default=1e-3)
    parser.add_argument("--subject_column", default=None,
                        help="Optional CSV column containing subject IDs. If omitted, IDs are "
                             "derived from the image path like encode_latents.py.")
    parser.add_argument("--metadata_csv", default=None,
                        help="Optional metadata CSV used with --group_by_label")
    parser.add_argument("--metadata_subject_column", default="Unnamed: 0",
                        help="Subject-code column in --metadata_csv")
    parser.add_argument("--metadata_label_column", default="DISEASE",
                        help="Label column in --metadata_csv, e.g. DISEASE")
    parser.add_argument("--group_by_label", action="store_true",
                        help="Build one template per metadata label and a global fallback")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for --group_by_label. Defaults to "
                             "CARDIODIT_RUNS_DIR/temporal_alignment/dtw_templates")
    parser.add_argument("--min_group_size", type=int, default=2,
                        help="Minimum labelled subjects required to build a group template")
    return parser.parse_args()


def _subject_id_from_image_path(path: str) -> str:
    return Path(path).stem.replace(".nii", "")


def _metadata_subject_key(value) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    match = re.match(r"^0*(\d+)(?:\.0)?", text)
    if match:
        return str(int(match.group(1)))
    return text


def _label_filename(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    if not safe:
        raise ValueError(f"Cannot make template filename from empty label {label!r}")
    return f"{safe}.pt"


def _load_1d_tensor(path: Path) -> torch.Tensor:
    return load_1d_tensor(path, "descriptor", keys=DESCRIPTOR_KEYS)


def _descriptor_path(sidecar_dir: Path, subject_id: str) -> Path:
    return find_descriptor_sidecar(sidecar_dir, subject_id)


def _keyframe_path(sidecar_dir: Path, keyframe_dir: Path | None, subject_id: str) -> Path:
    candidates = []
    if keyframe_dir is not None:
        candidates.append(keyframe_dir / f"{subject_id}.json")
    candidates.extend([
        sidecar_dir / "keyframes" / f"{subject_id}.json",
        sidecar_dir / f"{subject_id}.json",
    ])
    for path in candidates:
        if path.exists():
            return path
    tried = " and ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing keyframes for '{subject_id}': tried {tried}")


def _subject_ids(df: pd.DataFrame, subject_column: str | None) -> list[str]:
    if subject_column is not None:
        if subject_column not in df.columns:
            raise ValueError(f"CSV has no subject column '{subject_column}'")
        return [str(value) for value in df[subject_column].tolist()]
    if "image" not in df.columns:
        raise ValueError("CSV must contain an 'image' column unless --subject_column is set")
    return [_subject_id_from_image_path(str(path)) for path in df["image"].tolist()]


def _load_label_map(
    metadata_csv: str,
    subject_column: str,
    label_column: str,
) -> dict[str, str]:
    df = pd.read_csv(metadata_csv, low_memory=False)
    if subject_column not in df.columns:
        raise ValueError(f"Metadata CSV has no subject column '{subject_column}'")
    if label_column not in df.columns:
        raise ValueError(f"Metadata CSV has no label column '{label_column}'")

    mapping = {}
    for _, row in df.iterrows():
        key = _metadata_subject_key(row[subject_column])
        label = row[label_column]
        if not key or pd.isna(label):
            continue
        mapping[key] = str(label).strip()
    return mapping


def _build_one_template(
    subject_ids: list[str],
    descriptors: list[torch.Tensor],
    keyframes: list[dict],
    *,
    args,
    source_extra: dict,
) -> dict:
    template = build_dtw_template(
        descriptors,
        keyframes,
        T_out=args.target_frames,
        n_iters=args.n_iters,
        aggregation=args.aggregation,
        max_warp_fraction=args.max_warp_fraction,
        non_diagonal_penalty=args.non_diagonal_penalty,
    )
    template["source"] = {
        "csv": str(args.csv),
        "sidecar_dir": str(args.sidecar_dir),
        "keyframe_dir": str(args.keyframe_dir) if args.keyframe_dir else None,
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        **source_extra,
    }
    return template


def main():
    args = parse_args()
    sidecar_dir = Path(args.sidecar_dir)
    keyframe_dir = Path(args.keyframe_dir) if args.keyframe_dir else None
    if args.min_group_size < 1:
        raise ValueError(f"--min_group_size must be >= 1, got {args.min_group_size}")
    if args.group_by_label and args.metadata_csv is None:
        raise ValueError("--group_by_label requires --metadata_csv")

    df = pd.read_csv(args.csv)
    subject_ids = _subject_ids(df, args.subject_column)

    descriptors = []
    keyframes = []
    for subject_id in subject_ids:
        descriptor_path = _descriptor_path(sidecar_dir, subject_id)
        kf_path = _keyframe_path(sidecar_dir, keyframe_dir, subject_id)
        descriptors.append(_load_1d_tensor(descriptor_path))
        with open(kf_path) as f:
            keyframes.append(json.load(f))

    if not args.group_by_label:
        output = Path(args.output) if args.output else RUNS_ROOT / "temporal_alignment" / "dtw_template.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        template = _build_one_template(
            subject_ids,
            descriptors,
            keyframes,
            args=args,
            source_extra={"template_scope": "global"},
        )
        torch.save(template, output)
        print(f"Saved DTW template: {output}")
        print(f"n_subjects={len(subject_ids)}  n_frames={template['n_frames']}")
        print(f"keyframe_positions={template['keyframe_positions']}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else RUNS_ROOT / "temporal_alignment" / "dtw_templates"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = _load_label_map(
        args.metadata_csv,
        args.metadata_subject_column,
        args.metadata_label_column,
    )

    global_template = _build_one_template(
        subject_ids,
        descriptors,
        keyframes,
        args=args,
        source_extra={
            "template_scope": "global",
            "metadata_csv": str(args.metadata_csv),
            "metadata_label_column": args.metadata_label_column,
        },
    )
    torch.save(global_template, output_dir / "global.pt")
    print(f"Saved global fallback DTW template: {output_dir / 'global.pt'}")

    groups: dict[str, list[int]] = {}
    missing = []
    for idx, subject_id in enumerate(subject_ids):
        key = _metadata_subject_key(subject_id)
        label = label_map.get(key)
        if label is None:
            missing.append(subject_id)
            continue
        groups.setdefault(label, []).append(idx)

    if missing:
        raise ValueError(
            "Missing metadata labels for subjects: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )

    manifest = {
        "global_template": "global.pt",
        "metadata_csv": str(args.metadata_csv),
        "metadata_subject_column": args.metadata_subject_column,
        "metadata_label_column": args.metadata_label_column,
        "templates": {},
    }
    for label, indices in sorted(groups.items()):
        if len(indices) < args.min_group_size:
            print(
                f"Skipping label {label!r}: {len(indices)} subjects < "
                f"--min_group_size={args.min_group_size}"
            )
            continue
        group_subjects = [subject_ids[i] for i in indices]
        group_template = _build_one_template(
            group_subjects,
            [descriptors[i] for i in indices],
            [keyframes[i] for i in indices],
            args=args,
            source_extra={
                "template_scope": "label",
                "label": label,
                "metadata_csv": str(args.metadata_csv),
                "metadata_label_column": args.metadata_label_column,
            },
        )
        filename = _label_filename(label)
        torch.save(group_template, output_dir / filename)
        manifest["templates"][label] = filename
        print(f"Saved label DTW template: {output_dir / filename}  n={len(indices)}")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved DTW template manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

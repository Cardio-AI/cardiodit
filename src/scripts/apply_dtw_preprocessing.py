"""
Apply descriptor-DTW temporal alignment to preprocessed native-grid NIfTIs.

This script consumes ``data/preprocessed/MNM2_native_grid/manifest.csv`` and
writes image-space DTW-aligned 4D NIfTIs. It does not encode latents.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import RUNS_ROOT

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from src.utils.descriptor_io import (
    DESCRIPTOR_KEYS,
    TEMPLATE_KEYS,
    extract_1d_tensor,
    load_descriptor_sidecar,
)
from src.utils.temporal_align import align


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply descriptor-DTW alignment to native-grid preprocessed NIfTIs."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Native-grid manifest with subject_id, split, and preprocessed_image columns.",
    )
    parser.add_argument(
        "--native_root",
        required=True,
        help="Root directory for relative preprocessed_image paths in --manifest.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sidecar_dir", required=True)
    parser.add_argument("--keyframe_dir", required=True)
    parser.add_argument("--target_frames", type=int, default=32)
    parser.add_argument(
        "--template",
        default=None,
        help="Global DTW template .pt. Required unless --template_dir is set.",
    )
    parser.add_argument(
        "--template_dir",
        default=None,
        help="Pathology-specific template directory containing manifest.json.",
    )
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--metadata_subject_column", default="Unnamed: 0")
    parser.add_argument("--metadata_label_column", default="DISEASE")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist. Defaults to skip existing files.",
    )
    return parser.parse_args()


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


def load_metadata_label_map(
    metadata_csv: Path,
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


def load_template_manifest(template_dir: Path) -> dict:
    manifest_path = template_dir / "manifest.json"
    if not manifest_path.exists():
        return {"global_template": "global.pt", "templates": {}}
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.setdefault("global_template", "global.pt")
    manifest.setdefault("templates", {})
    return manifest


def resolve_label_template_path(template_dir: Path, manifest: dict, label: str) -> Path:
    filename = manifest.get("templates", {}).get(label, _label_filename(label))
    path = template_dir / filename
    if path.exists():
        return path
    fallback = template_dir / manifest.get("global_template", "global.pt")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Missing DTW template for label '{label}': tried {path} and fallback {fallback}"
    )


def load_template_file(path: Path, target_frames: int) -> tuple[torch.Tensor, dict | None, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    template = extract_1d_tensor(payload, path, "DTW template", keys=TEMPLATE_KEYS)
    if template.numel() != target_frames:
        raise ValueError(
            f"DTW template length mismatch: {path} has length {template.numel()}, "
            f"but --target_frames={target_frames}"
        )

    keyframe_positions = None
    if isinstance(payload, dict) and "keyframe_positions" in payload:
        keyframe_positions = {
            str(key): float(value)
            for key, value in payload["keyframe_positions"].items()
        }

    constraints = {}
    if isinstance(payload, dict) and isinstance(payload.get("dtw_constraints"), dict):
        for key in ("max_warp_fraction", "non_diagonal_penalty"):
            if key in payload["dtw_constraints"]:
                constraints[key] = float(payload["dtw_constraints"][key])
    return template, keyframe_positions, constraints


def load_keyframes(keyframe_dir: Path, subject_id: str) -> dict:
    path = keyframe_dir / f"{subject_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing keyframes: {path}")
    with open(path) as f:
        return json.load(f)


def save_like_input(path: Path, volume: np.ndarray, source_img: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = source_img.header.copy()
    out = nib.Nifti1Image(volume.astype(np.float32, copy=False), source_img.affine, header=header)
    qcode = int(source_img.header["qform_code"])
    scode = int(source_img.header["sform_code"])
    if qcode > 0:
        out.set_qform(source_img.get_qform(), code=qcode)
    if scode > 0:
        out.set_sform(source_img.get_sform(), code=scode)
    if qcode <= 0 and scode <= 0:
        out.set_qform(source_img.affine, code=1)
        out.set_sform(source_img.affine, code=1)
    out.header.set_data_dtype(np.float32)
    nib.save(out, path)


def align_one(
    native_path: Path,
    subject_id: str,
    *,
    sidecar_dir: Path,
    keyframe_dir: Path,
    template: torch.Tensor,
    keyframe_positions: dict | None,
    constraints: dict,
    target_frames: int,
) -> tuple[np.ndarray, nib.Nifti1Image, Path]:
    img = nib.load(native_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"{native_path}: expected 4D NIfTI, got shape {data.shape}")

    descriptor, descriptor_path = load_descriptor_sidecar(
        sidecar_dir,
        subject_id,
        "DTW descriptor",
    )
    if descriptor.numel() != data.shape[-1]:
        raise ValueError(
            f"Descriptor length mismatch for {subject_id}: {descriptor_path} has "
            f"{descriptor.numel()} values but {native_path} has T={data.shape[-1]}"
        )

    sidecar = {
        "descriptor": descriptor,
        "template": template,
        **constraints,
    }
    if keyframe_positions is not None:
        sidecar["keyframes"] = load_keyframes(keyframe_dir, subject_id)
        sidecar["template_keyframe_positions"] = keyframe_positions

    tensor = torch.from_numpy(data).unsqueeze(0)
    aligned = align(tensor, method="dtw", T_out=target_frames, **sidecar)
    return aligned.squeeze(0).cpu().numpy().astype(np.float32, copy=False), img, descriptor_path


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_image_csv(path: Path, rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image": str(output_root / row["preprocessed_image"])})


def main():
    args = parse_args()
    native_root = Path(args.native_root)
    output_root = Path(args.output_root).expanduser().resolve()
    sidecar_dir = Path(args.sidecar_dir)
    keyframe_dir = Path(args.keyframe_dir)

    if bool(args.template) == bool(args.template_dir):
        raise ValueError("Provide exactly one of --template or --template_dir")
    if args.template_dir and args.metadata_csv is None:
        raise ValueError("--template_dir requires --metadata_csv")

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    template_cache: dict[Path, tuple[torch.Tensor, dict | None, dict]] = {}
    label_map = None
    template_manifest = None
    template_dir = None

    global_template = None
    global_keyframe_positions = None
    global_constraints = None
    if args.template:
        global_template, global_keyframe_positions, global_constraints = load_template_file(
            Path(args.template),
            args.target_frames,
        )
    else:
        template_dir = Path(args.template_dir)
        template_manifest = load_template_manifest(template_dir)
        label_map = load_metadata_label_map(
            Path(args.metadata_csv),
            args.metadata_subject_column,
            args.metadata_label_column,
        )

    out_rows = []
    split_rows: dict[str, list[dict]] = {}
    for idx, row in enumerate(rows, start=1):
        subject_id = row["subject_id"]
        split = row["split"]
        native_path = native_root / row["preprocessed_image"]
        out_rel = Path(split) / f"{subject_id}.nii.gz"
        out_path = output_root / out_rel

        label = ""
        template_path = Path(args.template) if args.template else None
        template = global_template
        keyframe_positions = global_keyframe_positions
        constraints = global_constraints or {}
        if template_dir is not None:
            label = label_map.get(_metadata_subject_key(subject_id), "")
            if not label:
                raise ValueError(f"Missing metadata label for subject '{subject_id}'")
            template_path = resolve_label_template_path(template_dir, template_manifest, label)
            if template_path not in template_cache:
                template_cache[template_path] = load_template_file(template_path, args.target_frames)
            template, keyframe_positions, constraints = template_cache[template_path]

        if args.overwrite or not out_path.exists():
            aligned, source_img, descriptor_path = align_one(
                native_path,
                subject_id,
                sidecar_dir=sidecar_dir,
                keyframe_dir=keyframe_dir,
                template=template,
                keyframe_positions=keyframe_positions,
                constraints=constraints,
                target_frames=args.target_frames,
            )
            save_like_input(out_path, aligned, source_img)
            shape = "x".join(str(v) for v in aligned.shape)
            T_native = int(source_img.shape[-1])
            action = "saved"
        else:
            img = nib.load(out_path)
            descriptor_path = sidecar_dir / "motion_descriptor" / f"{subject_id}.npy"
            shape = "x".join(str(v) for v in img.shape)
            T_native = int(row.get("T_native") or 0)
            action = "skipped"

        manifest_row = {
            "subject_id": subject_id,
            "split": split,
            "source_image": row.get("source_image", ""),
            "native_preprocessed_image": row["preprocessed_image"],
            "preprocessed_image": str(out_rel),
            "descriptor": str(descriptor_path),
            "template": str(template_path),
            "label": label,
            "shape": shape,
            "T_native": T_native,
            "T_out": args.target_frames,
            "max_warp_fraction": constraints.get("max_warp_fraction", ""),
            "non_diagonal_penalty": constraints.get("non_diagonal_penalty", ""),
        }
        out_rows.append(manifest_row)
        split_rows.setdefault(split, []).append(manifest_row)

        if idx == 1 or idx % 25 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] {action} {out_path}")

    write_manifest(output_root / "manifest.csv", out_rows)
    for split, split_manifest_rows in split_rows.items():
        write_manifest(output_root / f"{split}_manifest.csv", split_manifest_rows)
        write_image_csv(output_root / f"{split}.csv", split_manifest_rows, output_root)
    print(f"Saved manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()

"""
Experimental temporal-alignment encoder preserved from CardioDiT-TempAlign.

Writes historical bare-tensor latents, not the main encoder's provenance contract.
Use training.allow_legacy_latents=true explicitly when training with these files.
Prefer encode_latents.py for new contracted datasets. Linear/Fourier/piecewise/
DTW methods intentionally retain normalize/crop-then-align semantics.

The VQ-GAN is a 3D model operating on individual 2D+t slices (H, W, T).
Each full CMR volume (D, H, W, T) is encoded slice-by-slice along the depth
axis and the resulting latents are stacked to form the 4D latent tensor
(C, D, H/f, W/f, T/ft) saved as a .pt file.

Temporal alignment for variable-length cines is handled in image space before
slicing into the VQ-GAN; see ``src/utils/temporal_align.py`` for available
methods.
"""

import argparse
import json
from pathlib import Path
import re
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    CenterSpatialCropd,
    SpatialPadd,
)
from omegaconf import OmegaConf

from src.utils.stage1_loading import load_stage1_strict
from src.data.dataloading import PermuteDimensionsd, UnsqueezeChanneld  # noqa: F401
from src.utils.descriptor_io import (
    DESCRIPTOR_KEYS,
    TEMPLATE_KEYS,
    descriptor_sidecar_candidates,
    extract_1d_tensor,
    load_1d_tensor,
)
from src.utils.temporal_align import align


ALIGN_METHODS = ("cyclic", "linear", "fourier", "piecewise", "dtw", "motionfield", "native")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode 4D CMR volumes into VQ-GAN latents")
    parser.add_argument("--csv", required=True, help="CSV with 'image' column (paths to 4D .nii.gz files)")
    parser.add_argument("--output_dir", required=True, help="Directory to store .pt latent files")
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to VQ-GAN checkpoint")
    parser.add_argument("--config", required=True, help="Path to VQ-GAN config yaml")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[224, 224, 32],
                        metavar=("H", "W", "T"), help="Crop size for each 2D+t slice")
    parser.add_argument("--target_frames", type=int, default=32,
                        help="Target temporal frames after alignment (ignored for 'native')")
    parser.add_argument("--target_z", type=int, default=10,
                        help="Z slices after center crop + black padding")
    parser.add_argument("--temporal_alignment", default="cyclic", choices=ALIGN_METHODS,
                        help="Image-space alignment strategy applied before VQ-GAN encoding")
    parser.add_argument("--native_t_multiple", type=int, default=8,
                        help="For --temporal_alignment=native: pad image T up to a multiple "
                             "of this value. Must equal vqgan_temporal_stride * DiT patch_size_t "
                             "so the resulting latent T is divisible by the DiT temporal patch. "
                             "Default 8 matches the canonical config (stride 4, pt 2).")
    parser.add_argument("--sidecar_dir", default=None,
                        help="Directory of per-subject sidecar files "
                             "(phi_t/alpha_t/descriptors/motion_descriptor/keyframes) "
                             "from Mueller et al. pipeline. "
                             "Required for piecewise/dtw/motionfield, optional for native "
                             "(forwards alpha_t). For piecewise, this may point either to the "
                             "sidecar root or directly to the keyframes dir. For dtw, expected "
                             "files are dtw_template.pt and 1D .pt or .npy descriptors under "
                             "alpha_t/, descriptors/, or motion_descriptor/.")
    parser.add_argument("--keyframe_dir", default=None,
                        help="Optional directory of <subject_id>.json keyframes. Used by "
                             "piecewise and by dtw when dtw_template.pt contains "
                             "keyframe_positions. If omitted, sidecar_dir/keyframes/ and "
                             "sidecar_dir/ are tried.")
    parser.add_argument("--dtw_template_dir", default=None,
                        help="Optional directory of pathology-specific DTW templates. "
                             "Expected files are manifest.json, global.pt, and one "
                             "<label>.pt per metadata label. Overrides sidecar_dir/dtw_template.pt.")
    parser.add_argument("--metadata_csv", default=None,
                        help="Metadata CSV used with --dtw_template_dir to map subjects to labels")
    parser.add_argument("--metadata_subject_column", default="Unnamed: 0",
                        help="Subject-code column in --metadata_csv")
    parser.add_argument("--metadata_label_column", default="DISEASE",
                        help="Label column in --metadata_csv, e.g. DISEASE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dim_perm", type=int, nargs="+", default=[2, 3, 1, 0],
        metavar="I",
        help="Permutation applied to the 4-D tensor produced by EnsureChannelFirstd. "
             "For MNM2, MONAI loads (Z,H,W,T) and EnsureChannelFirstd returns "
             "(T,Z,H,W); the default (2,3,1,0) reorders to (H,W,Z,T). "
             "A channel dim is then inserted automatically. "
             "Override only if your NIfTI axis order differs.",
    )
    return parser.parse_args()


def load_model(config_path, ckpt_path, device):
    model, _, _ = load_stage1_strict(config_path, ckpt_path, device)
    return model


def build_transforms(roi_size, target_z=10, dim_perm=(0, 1, 2, 3, 4)):
    """
    Preprocessing for a full 4D CMR volume up to (but not including) temporal
    alignment.  Temporal alignment is applied as a separate step on the tensor
    returned from this pipeline so that the strategy can be selected per run.
    """
    H, W, _ = roi_size
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),                              # MNM2: (T,Z,H,W)
        PermuteDimensionsd(keys=["image"], perm=tuple(dim_perm)),         # (H,W,D,T)
        UnsqueezeChanneld(keys=["image"], dim=0),                         # (1,H,W,D,T)
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        CenterSpatialCropd(keys=["image"], roi_size=(H, W, target_z, -1)),
        SpatialPadd(keys=["image"], spatial_size=(H, W, target_z, -1), constant_values=-1.0),
        ToTensord(keys=["image"]),
    ])


def _keyframe_path(sidecar_dir: Path, subject_id: str, keyframe_dir: Path | None = None) -> Path:
    candidates = []
    if keyframe_dir is not None:
        candidates.append(keyframe_dir / f"{subject_id}.json")
    candidates.extend([
        sidecar_dir / f"{subject_id}.json",
        sidecar_dir / "keyframes" / f"{subject_id}.json",
    ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing keyframes sidecar: tried "
        + " and ".join(str(path) for path in candidates)
    )


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


def _extract_1d_tensor(payload, path: Path, kind: str) -> torch.Tensor:
    return extract_1d_tensor(payload, path, kind, keys=TEMPLATE_KEYS)


def load_dtw_template_file(path: Path, target_frames: int) -> tuple[torch.Tensor, dict | None, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing DTW template sidecar: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    template = _extract_1d_tensor(
        payload,
        path,
        "DTW template",
    )
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
    dtw_constraints = {}
    if isinstance(payload, dict) and isinstance(payload.get("dtw_constraints"), dict):
        constraints = payload["dtw_constraints"]
        for key in ("max_warp_fraction", "non_diagonal_penalty"):
            if key in constraints:
                dtw_constraints[key] = float(constraints[key])
    return template, keyframe_positions, dtw_constraints


def load_dtw_template(sidecar_dir: Path, target_frames: int) -> tuple[torch.Tensor, dict | None, dict]:
    return load_dtw_template_file(Path(sidecar_dir) / "dtw_template.pt", target_frames)


def load_dtw_template_manifest(template_dir: Path) -> dict:
    manifest_path = Path(template_dir) / "manifest.json"
    if not manifest_path.exists():
        return {"global_template": "global.pt", "templates": {}}
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.setdefault("global_template", "global.pt")
    manifest.setdefault("templates", {})
    return manifest


def resolve_label_template_path(template_dir: Path, manifest: dict, label: str) -> Path:
    templates = manifest.get("templates", {})
    filename = templates.get(label, _label_filename(label))
    path = Path(template_dir) / filename
    if path.exists():
        return path
    fallback = Path(template_dir) / manifest.get("global_template", "global.pt")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Missing DTW template for label '{label}': tried {path} and fallback {fallback}"
    )


def load_dtw_descriptor(sidecar_dir: Path, subject_id: str) -> tuple[torch.Tensor, Path]:
    root = Path(sidecar_dir)
    candidates = descriptor_sidecar_candidates(root, subject_id)
    for path in candidates:
        if path.exists():
            return load_1d_tensor(path, "DTW descriptor", keys=DESCRIPTOR_KEYS), path
    tried = " and ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing DTW descriptor sidecar for subject '{subject_id}': tried {tried}"
    )


def load_sidecar(
    sidecar_dir: Path,
    subject_id: str,
    method: str,
    *,
    keyframe_dir: Path | None = None,
    require_dtw_keyframes: bool = False,
) -> dict:
    """Load per-subject sidecar payload required by ``method``.

    Piecewise supports either a keyframe directory or a sidecar root containing
    ``keyframes/``. DTW expects descriptor sidecars as .pt or .npy under
    ``alpha_t/``, ``descriptors/``, or ``motion_descriptor/``. Motion-field
    tensor layout is still pending.
    """
    if sidecar_dir is None:
        return {}
    sidecar_dir = Path(sidecar_dir)
    payload = {}
    if method == "piecewise":
        kf_path = _keyframe_path(sidecar_dir, subject_id, keyframe_dir)
        with open(kf_path) as f:
            payload["keyframes"] = json.load(f)
    elif method == "dtw":
        descriptor, descriptor_path = load_dtw_descriptor(sidecar_dir, subject_id)
        payload["descriptor"] = descriptor
        payload["_descriptor_path"] = descriptor_path
        if require_dtw_keyframes:
            kf_path = _keyframe_path(sidecar_dir, subject_id, keyframe_dir)
            with open(kf_path) as f:
                payload["keyframes"] = json.load(f)
    elif method == "motionfield":
        phi_path = sidecar_dir / "phi_t" / f"{subject_id}.pt"
        if not phi_path.exists():
            raise FileNotFoundError(f"Missing phi_t sidecar: {phi_path}")
        payload["phi"] = torch.load(phi_path, map_location="cpu", weights_only=True)
    return payload


def load_alpha_t(sidecar_dir: Path, subject_id: str) -> torch.Tensor | None:
    if sidecar_dir is None:
        return None
    for path in descriptor_sidecar_candidates(sidecar_dir, subject_id):
        if path.exists():
            return load_1d_tensor(path, "alpha_t", keys=DESCRIPTOR_KEYS)
    return None


@torch.no_grad()
def encode_volume(model, volume_4d, device, batch_size=1):
    """
    Encode a full (1, H, W, D, T) CMR volume slice-by-slice along D.

    Returns a tensor of shape (C, D, h, w, t) where h=H/f, w=W/f, t=T/ft.
    """
    D = volume_4d.shape[3]
    latents = []

    batch_size = max(1, int(batch_size))
    for start in range(0, D, batch_size):
        slices = [
            volume_4d[:, :, :, d, :].unsqueeze(0)
            for d in range(start, min(start + batch_size, D))
        ]
        batch = torch.cat(slices, dim=0).to(device)  # (B,1,H,W,T)
        z = model.encode_stage_2_inputs(batch, quantized=True)
        latents.extend(z.cpu())

    return torch.stack(latents, dim=1)


def main():
    args = parse_args()
    print("Experimental aligned encoder: writing legacy bare-tensor latents; "
          "use training.allow_legacy_latents=true explicitly.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_dir = Path(args.sidecar_dir) if args.sidecar_dir else None
    keyframe_dir = Path(args.keyframe_dir) if args.keyframe_dir else None
    if args.temporal_alignment in ("piecewise", "dtw", "motionfield") and sidecar_dir is None:
        raise ValueError(
            f"--temporal_alignment={args.temporal_alignment} requires --sidecar_dir"
        )

    dtw_template = None
    dtw_template_keyframe_positions = None
    dtw_template_constraints = {}
    dtw_template_dir = Path(args.dtw_template_dir) if args.dtw_template_dir else None
    dtw_template_manifest = None
    dtw_template_cache: dict[Path, tuple[torch.Tensor, dict | None, dict]] = {}
    metadata_label_map = None
    if args.temporal_alignment == "dtw":
        if dtw_template_dir is not None:
            if args.metadata_csv is None:
                raise ValueError("--dtw_template_dir requires --metadata_csv")
            dtw_template_manifest = load_dtw_template_manifest(dtw_template_dir)
            metadata_label_map = load_metadata_label_map(
                Path(args.metadata_csv),
                args.metadata_subject_column,
                args.metadata_label_column,
            )
        else:
            dtw_template, dtw_template_keyframe_positions, dtw_template_constraints = load_dtw_template(
                sidecar_dir,
                args.target_frames,
            )

    model = load_model(args.config, args.vqvae_ckpt, device)
    transforms = build_transforms(args.roi_size, args.target_z, args.dim_perm)

    df = pd.read_csv(args.csv)
    image_paths = [str(row["image"]) for _, row in df.iterrows()]

    output_rows = []
    t_latents = []
    for img_path in image_paths:
        stem = Path(img_path).stem.replace(".nii", "")
        out_path = output_dir / f"{stem}.pt"

        if out_path.exists():
            print(f"Skipping {out_path} (exists)")
            z = torch.load(out_path, map_location="cpu", weights_only=True)
            t_latents.append(int(z.shape[-1]))
            output_rows.append(str(out_path))
            continue

        data = transforms({"image": img_path})
        volume = data["image"]                       # (1, H, W, D, T_in)
        subject_template = dtw_template
        subject_template_keyframe_positions = dtw_template_keyframe_positions
        subject_template_constraints = dtw_template_constraints
        if args.temporal_alignment == "dtw" and dtw_template_dir is not None:
            metadata_key = _metadata_subject_key(stem)
            label = metadata_label_map.get(metadata_key)
            if label is None:
                raise ValueError(
                    f"Missing metadata label for subject '{stem}' "
                    f"(metadata key '{metadata_key}')"
                )
            template_path = resolve_label_template_path(
                dtw_template_dir,
                dtw_template_manifest,
                label,
            )
            if template_path not in dtw_template_cache:
                dtw_template_cache[template_path] = load_dtw_template_file(
                    template_path,
                    args.target_frames,
                )
            (
                subject_template,
                subject_template_keyframe_positions,
                subject_template_constraints,
            ) = dtw_template_cache[template_path]

        sidecar = load_sidecar(
            sidecar_dir,
            stem,
            args.temporal_alignment,
            keyframe_dir=keyframe_dir,
            require_dtw_keyframes=subject_template_keyframe_positions is not None,
        )
        if args.temporal_alignment == "dtw":
            descriptor_path = sidecar.pop("_descriptor_path")
            descriptor = sidecar["descriptor"]
            if descriptor.numel() != volume.shape[-1]:
                raise ValueError(
                    f"DTW descriptor length mismatch for subject '{stem}': "
                    f"{descriptor_path} has length {descriptor.numel()}, "
                    f"but transformed image T={volume.shape[-1]}"
                )
            sidecar["template"] = subject_template
            if subject_template_keyframe_positions is not None:
                sidecar["template_keyframe_positions"] = subject_template_keyframe_positions
            sidecar.update(subject_template_constraints)
        volume = align(
            volume, method=args.temporal_alignment,
            T_out=args.target_frames, t_multiple=args.native_t_multiple,
            **sidecar,
        )                                            # (1, H, W, D, T_out)

        latent = encode_volume(model, volume, device, args.batch_size)  # (C, D, h, w, t)
        torch.save(latent, out_path)
        t_latents.append(int(latent.shape[-1]))
        print(f"Saved {out_path}  shape={list(latent.shape)}")

        if args.temporal_alignment == "native" and sidecar_dir is not None:
            alpha = load_alpha_t(sidecar_dir, stem)
            if alpha is not None:
                torch.save(alpha, output_dir / f"{stem}.alpha.pt")

        output_rows.append(str(out_path))

    out_csv = output_dir / "latents.csv"
    pd.DataFrame({"image": output_rows, "T_latent": t_latents}).to_csv(out_csv, index=False)
    print(f"\nSaved CSV → {out_csv}  ({len(output_rows)} entries)")


if __name__ == "__main__":
    main()

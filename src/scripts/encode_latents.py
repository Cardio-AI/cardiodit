"""Encode NIfTI cine volumes into self-describing Stage-1 latent payloads."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    import nibabel as nib
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        ScaleIntensityd,
        SpatialPadd,
        ToTensord,
    )
    from omegaconf import OmegaConf

    from src.data.dataloading import (
        CyclicPadTimed,
        MarkTemporalValidityd,
        PadTimeToMultipleD,
        PermuteDimensionsd,
        RecordDepthProvenanced,
        UnsqueezeChanneld,
        VQGAN_TRANSFORM_VERSION,
    )
    from src.data.latent_contract import (
        LatentContractError,
        make_latent_payload,
        reuse_identity,
        validate_latent_payload,
    )
    from src.data.temporal_alignment import (
        linear_phase,
        load_descriptor_sidecar,
        pool_alpha_t,
        resolve_descriptor_path,
    )
    from src.utils.checkpointing import atomic_torch_save, sha256_file
    from src.utils.stage1_loading import load_stage1_strict
except ModuleNotFoundError:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        class _TorchHelpStub:
            @staticmethod
            def no_grad():
                return lambda fn: fn

        torch = _TorchHelpStub()
        nib = pd = F = OmegaConf = None
    else:
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV with an image column")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--roi_size", type=int, nargs=3, default=None, metavar=("H", "W", "T"),
        help="Explicit override; otherwise training.roi_size from Stage-1 config.",
    )
    parser.add_argument(
        "--target_frames", default=None,
        help="Explicit fixed-frame override or 'native'; otherwise Stage-1 config.",
    )
    parser.add_argument(
        "--no_cyclic_pad", action="store_true",
        help="Explicitly override the config temporal policy with native length.",
    )
    parser.add_argument(
        "--target_frame_multiple", type=int, default=None,
        help="Explicit override; otherwise training.time_pad_multiple.",
    )
    parser.add_argument(
        "--target_z", type=int, default=None,
        help="Explicit depth-policy override; otherwise encoding.target_z is required.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--no_quantize", action="store_true")
    quantization.add_argument("--quantized", action="store_true")
    parser.add_argument(
        "--dim_perm", type=int, nargs=4, default=None, metavar=("I", "J", "K", "L"),
        help="Explicit raw-to-(H,W,D,T) permutation override.",
    )
    parser.add_argument("--phase_dir", type=str, default=None)
    parser.add_argument(
        "--geometry_policy",
        choices=["require_consistent", "prefer_sform", "prefer_qform", "canonical"],
        default=None,
        help="Required explicitly or as encoding.geometry_policy in Stage-1 config.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Atomically replace an existing stale/invalid latent instead of failing.",
    )
    return parser.parse_args()


def _config_value(config, section, key, default=None):
    node = config.get(section, {})
    return node.get(key, default) if node is not None else default


def resolve_encoding_options(args, config) -> dict:
    """Resolve preprocessing from Stage-1 config, applying only explicit CLI overrides."""
    roi = args.roi_size or _config_value(config, "training", "roi_size")
    if roi is None or len(roi) != 3:
        raise ValueError("Stage-1 training.roi_size (H,W,T) is required")
    roi = tuple(int(value) for value in roi)

    configured_target = _config_value(config, "training", "target_frames", roi[-1])
    configured_multiple = _config_value(config, "training", "time_pad_multiple")
    if args.no_cyclic_pad:
        if args.target_frames is not None or args.target_frame_multiple is not None:
            raise ValueError(
                "--no_cyclic_pad cannot be combined with another temporal override"
            )
        target_frames = None
        frame_multiple = None
    elif args.target_frames is not None:
        if args.target_frame_multiple is not None:
            raise ValueError(
                "Set only one of --target_frames and --target_frame_multiple"
            )
        target_frames = (
            None
            if str(args.target_frames).lower() == "native"
            else int(args.target_frames)
        )
        frame_multiple = None
    elif args.target_frame_multiple is not None:
        target_frames = None
        frame_multiple = int(args.target_frame_multiple)
    else:
        target_frames = (
            None
            if configured_target is None or str(configured_target).lower() == "native"
            else int(configured_target)
        )
        frame_multiple = (
            None if configured_multiple is None else int(configured_multiple)
        )
    if target_frames is not None and frame_multiple is not None:
        raise ValueError("Resolved config selects both fixed target_frames and time padding multiple")

    target_z = args.target_z
    if target_z is None:
        target_z = _config_value(config, "encoding", "target_z")
    if target_z is None:
        raise ValueError(
            "Depth policy is ambiguous: set encoding.target_z in the Stage-1 config "
            "or pass explicit --target_z."
        )

    if args.no_quantize:
        quantized = False
    elif args.quantized:
        quantized = True
    else:
        quantized = _config_value(config, "encoding", "quantized")
        if quantized is None:
            raise ValueError(
                "Latent quantization is ambiguous: set encoding.quantized in the "
                "Stage-1 config or pass --quantized/--no_quantize."
            )

    if args.dim_perm is not None:
        dim_perm = tuple(int(value) for value in args.dim_perm)
        permutation_source = "cli_raw_to_hwdt"
    else:
        training_perm = _config_value(config, "training", "spatial_permute")
        training_perm = (0, 1, 2, 3) if training_perm is None else tuple(int(v) for v in training_perm)
        if len(training_perm) != 4 or sorted(training_perm) != [0, 1, 2, 3]:
            raise ValueError("training.spatial_permute must be a permutation of four axes")
        # Config permutation resolves raw input to (T,H,W,D); encoding then
        # moves T last to produce (H,W,D,T).
        dim_perm = tuple(training_perm[index] for index in (1, 2, 3, 0))
        permutation_source = "config_training_spatial_permute_then_time_last"
    if sorted(dim_perm) != [0, 1, 2, 3]:
        raise ValueError("Resolved dim_perm must be a permutation of four axes")

    geometry_policy = args.geometry_policy or _config_value(config, "encoding", "geometry_policy")
    if geometry_policy is None:
        raise ValueError(
            "NIfTI geometry policy is ambiguous: set encoding.geometry_policy or "
            "pass --geometry_policy."
        )
    phase_dir = args.phase_dir or _config_value(config, "encoding", "phase_dir")
    phase_endpoint = bool(_config_value(config, "encoding", "phase_endpoint", True))
    return {
        "roi_size": roi,
        "target_frames": target_frames,
        "target_frame_multiple": frame_multiple,
        "target_z": int(target_z),
        "dim_perm": dim_perm,
        "permutation_source": permutation_source,
        "quantized": bool(quantized),
        "geometry_policy": str(geometry_policy),
        "phase_dir": phase_dir,
        # Preserve the existing inclusive endpoint unless a versioned config
        # explicitly opts into cyclic [0,1) semantics.
        "phase_endpoint": phase_endpoint,
    }


def load_model(config_path, ckpt_path, device):
    """Backward-compatible wrapper around the centralized strict loader."""
    model, _, _ = load_stage1_strict(config_path, ckpt_path, device)
    return model


def build_transforms(
    roi_size,
    target_frames,
    target_z,
    dim_perm,
    target_frame_multiple=None,
):
    """Build raw-to-(1,H,W,D,T) transforms with exact time/depth provenance."""
    height, width, _ = roi_size
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        PermuteDimensionsd(keys=["image"], perm=tuple(dim_perm)),
        UnsqueezeChanneld(keys=["image"], dim=0),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
    ]
    if target_frames is not None:
        transforms.append(CyclicPadTimed(keys=["image"], target_frames=target_frames))
    elif target_frame_multiple is not None:
        transforms.append(
            PadTimeToMultipleD(keys=["image"], multiple=target_frame_multiple)
        )
    else:
        transforms.append(MarkTemporalValidityd(keys=["image"], dim=-1))
    transforms.extend(
        [
            RecordDepthProvenanced(keys=["image"], target_depth=target_z, dim=-2),
            CenterSpatialCropd(keys=["image"], roi_size=(height, width, target_z, -1)),
            SpatialPadd(
                keys=["image"],
                spatial_size=(height, width, target_z, -1),
                constant_values=-1.0,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    return Compose(transforms)


@torch.no_grad()
def encode_volume(model, volume_4d, device, batch_size=1, quantized=True):
    """Encode one ``(1,H,W,D,T)`` volume slice-wise to ``(C,D,h,w,t)``."""
    depth = volume_4d.shape[3]
    latent_batches = []
    for start in range(0, depth, max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), depth)
        slices = volume_4d[:, :, :, start:end, :].permute(3, 0, 1, 2, 4).to(device)
        latent_batches.append(
            model.encode_stage_2_inputs(slices, quantized=quantized).cpu()
        )
    return torch.cat(latent_batches, dim=0).permute(1, 0, 2, 3, 4).contiguous()


def resolve_nifti_geometry(path: str | Path, policy: str) -> dict:
    """Validate qform/sform and return the explicitly selected geometry."""
    image = nib.load(str(path))
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    qcode, scode = int(qcode), int(scode)
    conflict = qcode > 0 and scode > 0 and not bool(
        torch.allclose(torch.as_tensor(qform), torch.as_tensor(sform), atol=1e-5, rtol=1e-5)
    )
    if policy == "require_consistent" and conflict:
        raise ValueError(
            f"NIfTI qform/sform conflict for {path}; choose prefer_qform, "
            "prefer_sform, or canonical explicitly."
        )
    if policy == "prefer_qform":
        if qcode <= 0:
            raise ValueError(f"NIfTI has no valid qform: {path}")
        affine, source = qform, "qform"
    elif policy in ("prefer_sform", "require_consistent"):
        if scode > 0:
            affine, source = sform, "sform"
        elif qcode > 0:
            affine, source = qform, "qform"
        else:
            raise ValueError(f"NIfTI has neither a coded qform nor sform: {path}")
    elif policy == "canonical":
        affine, source = torch.eye(4).numpy(), "canonical_synthetic"
    else:
        raise ValueError(f"Unknown geometry policy {policy!r}")
    return {
        "policy": policy,
        "selected": source,
        "qform_code": qcode,
        "sform_code": scode,
        "qform_sform_conflict": conflict,
        "affine": torch.as_tensor(affine).tolist(),
        "zooms": [float(value) for value in image.header.get_zooms()],
        "original_shape": [int(value) for value in image.shape],
    }


def _source_path(value, manifest_path: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (manifest_path.parent / expanded).resolve()


def _atomic_csv_save(frame, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _descriptor_contract(
    image_path,
    phase_dir,
    original_length,
    temporal_map,
    latent_length,
    phase_endpoint=True,
):
    descriptor_path = resolve_descriptor_path(image_path, phase_dir)
    descriptor_identity = None
    if descriptor_path is None:
        source_values = linear_phase(original_length, endpoint=phase_endpoint)
    else:
        descriptor_identity = {
            "name": descriptor_path.name,
            "sha256": sha256_file(descriptor_path),
            "size": descriptor_path.stat().st_size,
        }
        source_values = load_descriptor_sidecar(descriptor_path)
        source_values = pool_alpha_t(source_values, original_length, method="linear")
    transformed = source_values[temporal_map]
    latent_values = pool_alpha_t(transformed, latent_length, method="linear")
    return {
        "source": descriptor_identity,
        "source_frame_values": source_values.tolist(),
        "transformed_frame_values": transformed.tolist(),
        "latent_values": latent_values.tolist(),
        "interpolation": "linear_by_descriptor_value",
        "phase_endpoint": bool(phase_endpoint),
    }


def _latent_validity(frame_validity, latent_length):
    values = torch.as_tensor(frame_validity, dtype=torch.float32).view(1, 1, -1)
    pooled = F.adaptive_avg_pool1d(values, latent_length).view(-1)
    return (pooled >= 1.0 - 1e-7).tolist()


def _row_for_payload(path: Path, output_dir: Path, latent, contract):
    return {
        "image": os.path.relpath(path, output_dir),
        "source_image": contract["source"]["manifest_value"],
        "T_latent": int(latent.shape[-1]),
        "C_latent": int(latent.shape[0]),
        "Z_latent": int(latent.shape[1]),
        "X_latent": int(latent.shape[2]),
        "Y_latent": int(latent.shape[3]),
        "latent_quantized": bool(contract["quantized"]),
        "latent_scale": float(contract["scale"]["latent"]),
        "latent_schema_version": 1,
        "preprocessing_identity": contract["preprocessing_identity"],
        "stage1_checkpoint_sha256": contract["stage1"]["checkpoint_sha256"],
        "stage1_config_sha256": contract["stage1"]["config_sha256"],
        "source_sha256": contract["source"]["sha256"],
    }


def encode_manifest(args, model=None, config=None, stage1_identity=None):
    """Encode/verify every manifest row and atomically publish ``latents.csv``."""
    manifest_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config is None or stage1_identity is None or model is None:
        model, config, stage1_identity = load_stage1_strict(
            args.config, args.vqvae_ckpt, args.device
        )
    options = resolve_encoding_options(args, config)
    transforms = build_transforms(
        options["roi_size"],
        options["target_frames"],
        options["target_z"],
        options["dim_perm"],
        options["target_frame_multiple"],
    )
    frame = pd.read_csv(manifest_path)
    rows = []
    for _, input_row in frame.iterrows():
        manifest_value = str(input_row["image"])
        image_path = _source_path(manifest_value, manifest_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        source_hash = sha256_file(image_path)
        source = {
            "manifest_value": manifest_value,
            "sha256": source_hash,
            "size": image_path.stat().st_size,
        }
        geometry = resolve_nifti_geometry(image_path, options["geometry_policy"])

        # Resolve descriptor identity before reuse validation so a changed
        # sidecar invalidates an otherwise unchanged image latent.
        descriptor_path = resolve_descriptor_path(image_path, options["phase_dir"])
        descriptor_preprocessing = None
        if descriptor_path is not None:
            descriptor_preprocessing = {
                "name": descriptor_path.name,
                "sha256": sha256_file(descriptor_path),
                "size": descriptor_path.stat().st_size,
            }
        preprocessing = {
            **{key: value for key, value in options.items() if key != "phase_dir"},
            "transform_version": VQGAN_TRANSFORM_VERSION,
            "descriptor": descriptor_preprocessing,
        }
        identity_probe = {
            "source": source,
            "shapes": {},
            "index_maps": {},
            "validity_masks": {},
            "preprocessing": preprocessing,
            "spacing": geometry,
            "quantized": options["quantized"],
            "scale": {"latent": 1.0, "intensity_min": -1.0, "intensity_max": 1.0},
            "stage1": stage1_identity,
        }
        expected_identity = reuse_identity(identity_probe)
        stem = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
        output_path = output_dir / f"{stem}-{source_hash[:12]}.pt"

        if output_path.exists() and not args.force:
            try:
                existing = torch.load(output_path, map_location="cpu", weights_only=False)
                latent, contract = validate_latent_payload(
                    existing, expected_reuse_identity=expected_identity
                )
            except (OSError, RuntimeError, LatentContractError) as error:
                raise LatentContractError(
                    f"Existing latent is stale or incomplete: {output_path}. "
                    "Use --force to regenerate it."
                ) from error
            rows.append(_row_for_payload(output_path, output_dir, latent, contract))
            continue

        data = transforms({"image": str(image_path)})
        transformed = data["image"]
        latent = encode_volume(
            model,
            transformed,
            args.device,
            args.batch_size,
            quantized=options["quantized"],
        )
        temporal_map = torch.as_tensor(data["temporal_index_map"], dtype=torch.long)
        temporal_valid = torch.as_tensor(data["temporal_valid_mask"], dtype=torch.bool)
        depth_map = torch.as_tensor(data["depth_index_map"], dtype=torch.long)
        depth_valid = torch.as_tensor(data["depth_valid_mask"], dtype=torch.bool)
        descriptor = _descriptor_contract(
            image_path,
            options["phase_dir"],
            int(data["original_temporal_length"]),
            temporal_map,
            int(latent.shape[-1]),
            phase_endpoint=options["phase_endpoint"],
        )
        contract = {
            **identity_probe,
            "shapes": {
                "original_nifti": geometry["original_shape"],
                "transformed_image": [int(value) for value in transformed.shape],
                "latent": [int(value) for value in latent.shape],
            },
            "index_maps": {
                "temporal": temporal_map.tolist(),
                "depth": depth_map.tolist(),
            },
            "validity_masks": {
                "temporal": temporal_valid.tolist(),
                "depth": depth_valid.tolist(),
                "latent_temporal": _latent_validity(temporal_valid, latent.shape[-1]),
                "latent_depth": depth_valid.tolist(),
            },
            "descriptor": descriptor,
        }
        payload = make_latent_payload(latent, contract)
        atomic_torch_save(payload, output_path)
        verified_latent, verified_contract = validate_latent_payload(
            torch.load(output_path, map_location="cpu", weights_only=False),
            expected_reuse_identity=expected_identity,
        )
        rows.append(
            _row_for_payload(output_path, output_dir, verified_latent, verified_contract)
        )

    output_csv = output_dir / "latents.csv"
    _atomic_csv_save(pd.DataFrame(rows), output_csv)
    return output_csv, rows


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    output_csv, rows = encode_manifest(args)
    print(f"Saved verified CSV -> {output_csv} ({len(rows)} entries)")


if __name__ == "__main__":
    main()

"""
Unconditional sampling from a trained DiT4D latent diffusion model.

Generates 4D CMR volumes and saves them as NIfTI files. The reverse diffusion
process runs in the 4D latent space; the VQ-GAN decoder reconstructs each
depth slice independently and the results are stacked to form the full volume.
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from contextlib import nullcontext

try:
    import torch
    import numpy as np
    import nibabel as nib
    from torch import amp
    from omegaconf import OmegaConf

    from src.models.dit import DiT4D
    from src.models.ddpmscheduler import DDPMScheduler
    from src.models.ddimscheduler import DDIMScheduler
    from src.models.flow_matching_scheduler import FlowMatchingScheduler
    from src.models.dpm_solver_scheduler import DPMSolverPPScheduler
    from src.config_validation import validate_dit_config
    from src.data.temporal_alignment import load_descriptor_sidecar, pool_alpha_t
    from src.utils.checkpointing import atomic_json_save, atomic_torch_save
    from src.utils.checkpointing import (
        CheckpointMismatchError,
        adapt_legacy_checkpoint,
        canonical_hash,
        prepare_model_state_for_load,
        validate_checkpoint,
    )
    from src.utils.sample_integrity import (
        atomic_file_save,
        completion_matches,
        file_identity,
        flat_completion_matches,
        per_sample_seeds,
        publish_completion_manifest,
        publish_flat_completion_manifest,
        resolve_checkpoint_reference,
        software_versions,
    )
    from src.utils.stage1_loading import load_stage1_strict
except ModuleNotFoundError as exc:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        class _TorchHelpStub:
            @staticmethod
            def no_grad():
                return lambda fn: fn

        torch = _TorchHelpStub()
        np = nib = amp = OmegaConf = None
        DiT4D = DDPMScheduler = DDIMScheduler = None
        FlowMatchingScheduler = DPMSolverPPScheduler = None
        load_descriptor_sidecar = pool_alpha_t = load_stage1_strict = None
    else:
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained DiT4D LDM")
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--stage1_cfg", type=str, default=None)
    parser.add_argument("--diff_cfg", type=str, default=None)
    parser.add_argument("--diff_ckpt", type=str, required=True)
    parser.add_argument(
        "--frozen_manifest",
        type=str,
        default=None,
        help="Frozen input manifest supplying hashed Stage-1 config/checkpoint identities.",
    )
    parser.add_argument("--output_dir", type=str, default=str(Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser() / "samples"))
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of independent latent trajectories evaluated per DiT forward pass.",
    )
    parser.add_argument(
        "--per_sample_output_dirs",
        action="store_true",
        help=(
            "Write each sample to sample_NNN/ with its own completion manifest. "
            "This supports checkpoint-level model loading with sample-level resume."
        ),
    )
    parser.add_argument(
        "--flat_sample_files",
        action="store_true",
        help=(
            "With --per_sample_output_dirs and one decoder mode, write sample_NNN.nii.gz "
            "directly in --output_dir; keep metadata and manifests in hidden subdirectories."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument(
        "--scheduler", type=str, default="auto",
        choices=["auto", "ddpm", "ddim", "dpm_pp", "flow_matching"],
        help="'auto' uses flow_matching for flow-matching configs and ddpm otherwise.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help=(
            "Latent scale factor. Defaults to checkpoint scale_factor when "
            "present, otherwise diff_cfg.training.scale_factor."
        ),
    )
    parser.add_argument(
        "--skip_decoder_quantization",
        action="store_true",
        help="Legacy alias for --decoder_mode direct.",
    )
    parser.add_argument(
        "--decoder_mode",
        default="both",
        choices=["both", "direct", "quantized"],
        help="Decode each generated latent directly, quantized, or through both paths.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ema",
        choices=["ema", "raw", "both"],
        help="Which DiT weights to sample with. 'ema' uses checkpoint EMA shadow weights; "
             "'raw' uses the non-EMA model weights saved in the checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed; sample i uses seed + absolute sample index.",
    )
    parser.add_argument(
        "--latent_shape", type=int, nargs=5, default=None,
        metavar=("C", "D", "H", "W", "T"),
        help="Shape of one 4D latent, e.g. 8 10 28 28 8. "
             "Defaults to in_channels + model.params.input_size from --diff_cfg.",
    )
    parser.add_argument("--sample_index_start", type=int, default=0)
    parser.add_argument(
        "--geometry",
        choices=["canonical_synthetic", "template"],
        default="canonical_synthetic",
        help="Declare canonical synthetic geometry or copy an explicit NIfTI template.",
    )
    parser.add_argument("--template_nifti", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--T_latent", type=int, default=None,
        help="Override the latent temporal length used for sampling.",
    )
    parser.add_argument(
        "--T_image", type=int, default=None,
        help="Image-frame temporal length. Used with --vqgan_temporal_stride "
             "to infer T_latent when --latent_shape is omitted or overridden.",
    )
    parser.add_argument(
        "--vqgan_temporal_stride", type=int, default=None,
        help="Temporal downsampling stride of the VQ-GAN. If set with "
             "--T_image, T_latent=ceil(T_image/stride).",
    )
    parser.add_argument(
        "--alpha_t_path", type=str, default=None,
        help="Optional temporal descriptor sidecar passed to DiT4D.forward(alpha_t=...).",
    )
    parser.add_argument(
        "--amp_dtype", type=str, default="auto",
        choices=["auto", "none", "fp16", "float16", "bf16", "bfloat16"],
        help="Autocast dtype for DiT sampling. 'auto' uses bf16 on CUDA when supported.",
    )
    parser.add_argument(
        "--output_layout", type=str, default="4d",
        choices=["4d", "frames", "both"],
        help="Save one 4D NIfTI, separate 3D frame files, or both.",
    )
    parser.add_argument(
        "--debug_latents",
        action="store_true",
        help="Save the final generated latent tensor before VQ-GAN decoding.",
    )
    parser.add_argument(
        "--spacing", type=float, nargs=4, default=[10.0, 1.7, 1.7, 1.0],
        metavar=("D", "H", "W", "T"),
        help="Image-space voxel spacing: D,H,W in mm, T in frames. "
             "Written to the NIfTI affine (D,H,W) and pixdim (T).",
    )
    parser.add_argument("--output_axes", choices=["dhw", "hwd"], default="dhw")
    parser.add_argument("--flip_axes", type=int, nargs="*", default=[])
    parser.add_argument("--foreground_crop", action="store_true")
    parser.add_argument("--foreground_threshold", type=float, default=-0.95)
    parser.add_argument("--foreground_min_fraction", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_stage1(cfg_path, ckpt_path, device):
    model, _, _ = load_stage1_strict(cfg_path, ckpt_path, device)
    return model


def apply_frozen_manifest(args):
    if args.frozen_manifest is None:
        if args.stage1_cfg is None or args.stage1_ckpt is None:
            raise ValueError("Provide Stage-1 inputs or --frozen_manifest.")
        return None
    manifest_path = Path(args.frozen_manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs", {})
    for attribute, field in (
        ("stage1_cfg", "stage1_config"),
        ("stage1_ckpt", "stage1_checkpoint"),
        ("diff_ckpt", "diffusion_checkpoint"),
    ):
        identity = inputs.get(field)
        value = getattr(args, attribute) or (identity or {}).get("path")
        if value is None:
            raise ValueError(f"Frozen manifest is missing inputs.{field}.path")
        value = Path(value).expanduser()
        if not value.is_absolute():
            value = manifest_path.parent / value
        value = value.resolve()
        observed = file_identity(value)
        if identity is not None and observed["sha256"] != identity.get("sha256"):
            raise RuntimeError(f"Frozen manifest hash mismatch for {field}.")
        setattr(args, attribute, str(value))
    return payload


def load_sampling_config(cfg_path, checkpoint_path, checkpoint=None):
    if cfg_path is not None:
        return OmegaConf.load(cfg_path)
    if checkpoint is None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint = adapt_legacy_checkpoint(payload)
    if checkpoint.get("legacy") or checkpoint.get("resolved_config") is None:
        raise RuntimeError("Legacy DiT checkpoints require an explicit --diff_cfg.")
    validate_checkpoint(checkpoint)
    return OmegaConf.create(checkpoint["resolved_config"])


def load_dit(cfg_path, ckpt_path, device, weights="ema", checkpoint=None):
    if checkpoint is None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint = adapt_legacy_checkpoint(payload)
    modern = not checkpoint.get("legacy", False)
    if modern:
        validate_checkpoint(checkpoint)
    if cfg_path is None:
        if not modern or checkpoint.get("resolved_config") is None:
            raise RuntimeError("Legacy DiT checkpoints require an explicit --diff_cfg.")
        cfg = OmegaConf.create(checkpoint["resolved_config"])
        config_source = "checkpoint.resolved_config"
    else:
        cfg = OmegaConf.load(cfg_path)
        config_source = str(Path(cfg_path).resolve())
        if modern:
            resolved = OmegaConf.to_container(cfg, resolve=True)
            expected = checkpoint.get("provenance", {}).get("resolved_config", {}).get("sha256")
            actual = canonical_hash(resolved)
            if expected is not None and actual != expected:
                raise CheckpointMismatchError(
                    "External --diff_cfg does not match checkpoint resolved-config provenance."
                )

    model = DiT4D(**cfg.model.params)
    state_dict = dict(checkpoint["model"])
    if weights == "ema":
        ema = checkpoint.get("ema")
        if ema is None or not isinstance(ema.get("shadow"), dict):
            raise RuntimeError(f"Requested EMA weights, but checkpoint has no EMA state: {ckpt_path}")
        state_dict.update(ema["shadow"])
    migrated_checkpoint = dict(checkpoint)
    migrated_checkpoint["model"] = state_dict
    rope4d = str(cfg.model.params.get("pos_embed_mode", "")) == "rope4d"
    state_dict, migrations = prepare_model_state_for_load(
        migrated_checkpoint,
        set(model.state_dict()),
        allow_legacy_rope4d_pos_embed=rope4d,
    )
    for migration in migrations:
        warnings.warn(f"Applied explicit legacy migration: {migration}", RuntimeWarning)
    model.load_state_dict(state_dict, strict=True)

    latent_mean = checkpoint.get("latent_mean")
    latent_std = checkpoint.get("latent_std")
    if latent_mean is not None:
        latent_mean = latent_mean.to(device)
        latent_std = latent_std.to(device)

    metadata = {
        "scale_factor": checkpoint.get("scale_factor"),
        "normalize_latents": checkpoint.get("normalize_latents"),
        "checkpoint_schema": checkpoint.get("checkpoint_schema"),
        "config_source": config_source,
        "legacy_migrations": migrations,
    }

    return model.to(device).eval().requires_grad_(False), cfg, latent_mean, latent_std, metadata


def build_scheduler(scheduler_name, diff_cfg):
    cfg = dict(diff_cfg.scheduler)
    if scheduler_name == "ddpm":
        return DDPMScheduler(**cfg)
    elif scheduler_name == "ddim":
        return DDIMScheduler(**cfg)
    elif scheduler_name == "dpm_pp":
        return DPMSolverPPScheduler(**cfg)
    elif scheduler_name == "flow_matching":
        return FlowMatchingScheduler(**cfg)
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'")


def resolve_scheduler_name(requested, diff_cfg):
    requested = str(requested).lower()
    scheduler_type = str(diff_cfg.get("scheduler_type", "ddpm")).lower()
    is_flow_config = scheduler_type == "flow_matching"

    if requested == "auto":
        return "flow_matching" if is_flow_config else "ddpm"

    if requested == "flow_matching" and not is_flow_config:
        raise ValueError(
            "--scheduler flow_matching requires a DiT config with "
            "scheduler_type: flow_matching."
        )
    if requested != "flow_matching" and is_flow_config:
        raise ValueError(
            f"--scheduler {requested} is incompatible with scheduler_type: "
            "flow_matching. Use --scheduler auto or --scheduler flow_matching."
        )
    return requested


def resolve_scale_factor(cli_scale_factor, diff_cfg, ckpt_scale_factor=None):
    if cli_scale_factor is not None:
        if ckpt_scale_factor is not None and not math.isclose(
            float(cli_scale_factor), float(ckpt_scale_factor), rel_tol=1e-6, abs_tol=1e-8
        ):
            warnings.warn(
                "CLI --scale_factor does not match checkpoint scale_factor "
                f"({cli_scale_factor} vs {ckpt_scale_factor}); using CLI value.",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(cli_scale_factor)
    if ckpt_scale_factor is not None:
        return float(ckpt_scale_factor)
    return float(diff_cfg.training.get("scale_factor", 1.0))


def validate_normalization_metadata(diff_cfg, ckpt_metadata):
    ckpt_normalize = ckpt_metadata.get("normalize_latents")
    if ckpt_normalize is None:
        return
    cfg_normalize = bool(diff_cfg.training.get("normalize_latents", False))
    if bool(ckpt_normalize) != cfg_normalize:
        raise RuntimeError(
            "Checkpoint normalization metadata does not match --diff_cfg: "
            f"checkpoint normalize_latents={bool(ckpt_normalize)}, "
            f"config normalize_latents={cfg_normalize}."
        )


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def build_sample_metadata(
    args,
    diff_cfg,
    ckpt_metadata,
    scheduler_name,
    scale_factor,
    latent_shape,
    amp_dtype,
    latent_mean,
    scheduler=None,
    provenance=None,
    decoder_modes=None,
    sample_seeds=None,
):
    model_params = diff_cfg.model.params
    training_cfg = diff_cfg.training
    return {
        "stage1_cfg": args.stage1_cfg,
        "stage1_ckpt": args.stage1_ckpt,
        "diff_cfg": args.diff_cfg,
        "diff_ckpt": args.diff_ckpt,
        "weights": args.weights,
        "scheduler": scheduler_name,
        "timesteps": int(args.timesteps),
        "timestep_grid": (
            scheduler.timesteps.detach().cpu().tolist()
            if scheduler is not None
            else None
        ),
        "solver_order": int(getattr(scheduler, "solver_order", 1)),
        "seed": args.seed,
        "latent_shape": [int(v) for v in latent_shape],
        "scale_factor": float(scale_factor),
        "checkpoint_scale_factor": ckpt_metadata.get("scale_factor"),
        "config_scale_factor": float(training_cfg.get("scale_factor", 1.0)),
        "normalize_latents": bool(training_cfg.get("normalize_latents", False)),
        "checkpoint_normalize_latents": ckpt_metadata.get("normalize_latents"),
        "latent_stats_present": latent_mean is not None,
        "decoder_quantization": not args.skip_decoder_quantization,
        "decoder_modes": list(decoder_modes or resolve_decoder_modes(args)),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
        "output_layout": args.output_layout,
        "output_transform": {key: getattr(args, key, default) for key, default in (
            ("output_axes", "dhw"), ("flip_axes", []), ("foreground_crop", False),
            ("foreground_threshold", -0.95), ("foreground_min_fraction", 0.005),
        )},
        "spacing": [float(v) for v in args.spacing],
        "device": args.device,
        "n_samples": int(args.n_samples),
        "batch_size": int(getattr(args, "batch_size", 1)),
        "alpha_t_path": args.alpha_t_path,
        "per_sample_seeds": list(sample_seeds or []),
        "geometry": {
            "type": getattr(args, "geometry", "canonical_synthetic"),
            "template_nifti": getattr(args, "template_nifti", None),
        },
        "provenance": provenance,
        "model": {
            "input_size": [int(v) for v in model_params.input_size],
            "patch_size": [int(v) for v in model_params.patch_size],
            "pos_embed_mode": str(model_params.get("pos_embed_mode", model_params.get("temporal_pe", "absolute"))),
            "self_conditioning": bool(model_params.get("self_conditioning", False)),
            "variable_shape": bool(model_params.get("variable_shape", False)),
        },
    }


def write_sample_metadata(out_dir, metadata):
    metadata_path = out_dir / "sample_metadata.json"
    atomic_json_save(_jsonify(metadata), metadata_path)
    return metadata_path


def resolve_decoder_modes(args):
    if getattr(args, "skip_decoder_quantization", False):
        if getattr(args, "decoder_mode", "both") not in ("both", "direct"):
            raise ValueError("--skip_decoder_quantization conflicts with --decoder_mode quantized.")
        return ("direct",)
    mode = getattr(args, "decoder_mode", "both")
    return ("direct", "quantized") if mode == "both" else (mode,)


def resolve_latent_shape(args, diff_cfg):
    if args.latent_shape is not None:
        shape = list(args.latent_shape)
    else:
        params = diff_cfg.model.params
        shape = [int(params.in_channels), *[int(v) for v in params.input_size]]

    if args.T_latent is not None:
        shape[-1] = int(args.T_latent)
    elif args.T_image is not None and args.vqgan_temporal_stride is not None:
        if args.T_image <= 0 or args.vqgan_temporal_stride <= 0:
            raise ValueError("Image length and temporal stride must be positive")
        patch_t = int(diff_cfg.model.params.patch_size[-1])
        shape[-1] = int(math.ceil(args.T_image / (args.vqgan_temporal_stride * patch_t))) * patch_t
    return shape


def load_alpha_t(args, t_latent, device):
    if args.alpha_t_path is None:
        return None
    alpha = load_descriptor_sidecar(args.alpha_t_path)
    alpha = pool_alpha_t(alpha, t_latent, method="linear")
    return alpha.to(device=device, dtype=torch.float32).unsqueeze(0)


def resolve_amp_dtype(device, amp_dtype):
    amp_dtype = str(amp_dtype).lower()
    if amp_dtype == "none" or device.type == "cpu":
        return None
    if amp_dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if amp_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if amp_dtype in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown amp dtype '{amp_dtype}'")


def autocast_context(device, dtype):
    if dtype is None:
        return nullcontext()
    return amp.autocast(device_type=device.type, dtype=dtype)


def make_output_image(vol, spacing, geometry="canonical_synthetic", template_nifti=None, **export_options):
    if geometry == "template":
        if template_nifti is None:
            raise ValueError("--geometry template requires --template_nifti.")
        template = nib.load(str(template_nifti))
        expected_shape = tuple(template.shape if vol.ndim == 4 else template.shape[:3])
        if tuple(vol.shape) != expected_shape:
            raise ValueError(
                "Template geometry shape does not match generated output axes: "
                f"template={tuple(template.shape)}, output={tuple(vol.shape)}. "
                "Resample explicitly before attaching patient geometry."
            )
        header = template.header.copy()
        img = nib.Nifti1Image(vol.astype(np.float32), affine=template.affine, header=header)
    elif geometry == "canonical_synthetic":
        sp_d, sp_h, sp_w = spacing[:3]
        affine = np.diag([sp_d, sp_h, sp_w, 1.0]).astype(np.float64)
        img = nib.Nifti1Image(vol.astype(np.float32), affine=affine)
        if vol.ndim == 4:
            sp_t = spacing[3] if len(spacing) > 3 else 1.0
            img.header.set_zooms((sp_d, sp_h, sp_w, sp_t))
        else:
            img.header.set_zooms((sp_d, sp_h, sp_w))
        img.header["descrip"] = b"canonical synthetic geometry; not patient geometry"
    else:
        raise ValueError(f"Unknown geometry policy: {geometry}")
    return transform_output_image(img, **export_options)


def save_volume(vol, out_path, spacing, geometry="canonical_synthetic", template_nifti=None, **export_options):
    img = make_output_image(vol, spacing, geometry, template_nifti, **export_options)
    atomic_file_save(out_path, lambda temporary: nib.save(img, temporary), suffix=".nii.gz")


def transform_output_image(img, output_axes="dhw", flip_axes=(), foreground_crop=False,
                           foreground_threshold=-0.95, foreground_min_fraction=0.005):
    """Reorder/crop samples while retaining their voxel-to-world coordinates."""
    volume = np.asanyarray(img.dataobj)
    affine = img.affine.copy()
    temporal_zoom = img.header.get_zooms()[3] if volume.ndim == 4 else None
    if output_axes == "hwd":
        order = [1, 2, 0] + list(range(3, volume.ndim))
        volume = volume.transpose(order)
        index_map = np.eye(4)
        index_map[:3, :3] = np.eye(3)[:, [1, 2, 0]]
        affine = affine @ index_map
    elif output_axes != "dhw":
        raise ValueError(f"Unknown output_axes: {output_axes}")
    for axis in flip_axes:
        if axis not in (0, 1, 2):
            raise ValueError("flip_axes must contain only spatial axes 0, 1, 2")
        index_map = np.eye(4)
        index_map[axis, axis] = -1
        index_map[axis, 3] = volume.shape[axis] - 1
        affine = affine @ index_map
        volume = np.flip(volume, axis=axis)
    if foreground_crop:
        if not 0 <= foreground_min_fraction < 1:
            raise ValueError("foreground_min_fraction must be in [0, 1)")
        foreground = volume > foreground_threshold
        slices, starts = [], []
        for axis in range(3):
            profile = foreground.mean(axis=tuple(i for i in range(volume.ndim) if i != axis))
            keep = np.flatnonzero(profile > foreground_min_fraction)
            start, stop = (int(keep[0]), int(keep[-1]) + 1) if keep.size else (0, volume.shape[axis])
            slices.append(slice(start, stop))
            starts.append(start)
        volume = volume[tuple(slices) + (slice(None),) * (volume.ndim - 3)]
        affine[:3, 3] += affine[:3, :3] @ np.asarray(starts)
    if output_axes == "dhw" and not flip_axes and not foreground_crop:
        return img
    result = nib.Nifti1Image(volume.astype(np.float32), affine, header=img.header.copy())
    if temporal_zoom is not None:
        result.header.set_zooms((*result.header.get_zooms()[:3], temporal_zoom))
    result.set_qform(affine, code=1)
    result.set_sform(affine, code=1)
    return result


def save_output(
    vol,
    out_dir,
    sample_index,
    spacing,
    output_layout,
    geometry="canonical_synthetic",
    template_nifti=None,
    **export_options,
):
    # Compute one spatial field of view from the whole cine. In particular,
    # foreground motion must not give each exported frame a different crop.
    image = make_output_image(vol, spacing, geometry, template_nifti, **export_options)
    saved = []
    if output_layout in ("4d", "both"):
        out_path = out_dir / f"sample_{sample_index:03d}.nii.gz"
        atomic_file_save(out_path, lambda temporary: nib.save(image, temporary), suffix=".nii.gz")
        saved.append(out_path)

    if output_layout in ("frames", "both"):
        frame_dir = out_dir / f"sample_{sample_index:03d}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        volume = np.asanyarray(image.dataobj)
        for frame_idx in range(volume.shape[-1]):
            frame_path = frame_dir / f"frame_{frame_idx:03d}.nii.gz"
            frame = nib.Nifti1Image(
                volume[..., frame_idx], image.affine, header=image.header.copy(),
            )
            atomic_file_save(frame_path, lambda temporary: nib.save(frame, temporary), suffix=".nii.gz")
            saved.append(frame_path)
    return saved


@torch.no_grad()
def decode_latent(stage1, latent_4d, quantize_before_decode=True):
    """
    Decode a 4D latent (C, D, H, W, T) → (1, D, H_full, W_full, T_full).

    The VQ-GAN decoder is applied slice-by-slice along D and results are
    stacked to reconstruct the full spatial volume.
    """
    D = latent_4d.shape[1]
    slices = []
    for d in range(D):
        z = latent_4d[:, d, :, :, :].unsqueeze(0)    # (1, C, H, W, T)
        recon = stage1.decode_stage_2_outputs(
            z, quantize=quantize_before_decode
        )                                              # (1, 1, H_full, W_full, T_full)
        slices.append(recon[0])                        # (1, H_full, W_full, T_full)
    return torch.stack(slices, dim=1)                  # (1, D, H_full, W_full, T_full)


@torch.no_grad()
def decode_latent_modes(stage1, latent_4d, decoder_modes):
    """Decode one immutable latent through every requested Stage-1 path."""
    source = latent_4d.detach().clone()
    outputs = {}
    for mode in decoder_modes:
        if mode not in ("direct", "quantized"):
            raise ValueError(f"Unknown decoder mode: {mode}")
        outputs[mode] = decode_latent(
            stage1,
            source.clone(),
            quantize_before_decode=mode == "quantized",
        )
    return outputs


def _individual_sample_request(request, sample_index, sample_seed):
    """Return the request identity used by historical one-sample invocations."""
    individual = dict(request)
    individual["sample_indices"] = [int(sample_index)]
    individual["per_sample_seeds"] = [int(sample_seed)]
    return individual


def _scheduler_step_batch(scheduler, noise_pred, timestep, sample, generators):
    """Advance a batch while preserving independent DDPM RNG streams."""
    if isinstance(scheduler, DDPMScheduler):
        outputs = [
            scheduler.step(
                noise_pred[index:index + 1],
                timestep,
                sample[index:index + 1],
                generator=generators[index],
            )
            for index in range(sample.shape[0])
        ]
        return (
            torch.cat([item[0] for item in outputs], dim=0),
            torch.cat([item[1] for item in outputs], dim=0),
        )

    try:
        return scheduler.step(
            noise_pred,
            timestep,
            sample,
            generator=generators[0],
        )
    except TypeError:
        return scheduler.step(noise_pred, timestep, sample)


def main():
    args = parse_args()
    apply_frozen_manifest(args)
    args.stage1_ckpt = str(resolve_checkpoint_reference(args.stage1_ckpt))
    args.diff_ckpt = str(resolve_checkpoint_reference(args.diff_ckpt))
    if args.n_samples < 1 or args.sample_index_start < 0 or args.batch_size < 1:
        raise ValueError(
            "n_samples and batch_size must be positive and sample_index_start non-negative."
        )
    if args.geometry == "template" and args.template_nifti is None:
        raise ValueError("--geometry template requires --template_nifti.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = torch.load(
        args.diff_ckpt,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    sampling_checkpoint = adapt_legacy_checkpoint(checkpoint_payload)
    if not sampling_checkpoint.get("legacy"):
        validate_checkpoint(sampling_checkpoint)
    diff_cfg = load_sampling_config(args.diff_cfg, args.diff_ckpt, sampling_checkpoint)
    if not sampling_checkpoint.get("legacy") and args.diff_cfg is not None:
        expected_config_hash = (
            sampling_checkpoint.get("provenance", {})
            .get("resolved_config", {})
            .get("sha256")
        )
        observed_config_hash = canonical_hash(OmegaConf.to_container(diff_cfg, resolve=True))
        if expected_config_hash is not None and observed_config_hash != expected_config_hash:
            raise CheckpointMismatchError(
                "External --diff_cfg does not match checkpoint resolved-config provenance."
            )
    validate_dit_config(
        diff_cfg,
        legacy_checkpoint=sampling_checkpoint,
        allow_legacy_scaling=bool(sampling_checkpoint.get("legacy")),
    )
    scheduler_name = resolve_scheduler_name(args.scheduler, diff_cfg)
    scheduler = build_scheduler(scheduler_name, diff_cfg)
    scheduler.set_timesteps(args.timesteps)
    is_flow_matching = isinstance(scheduler, FlowMatchingScheduler)
    C, D, H, W, T = resolve_latent_shape(args, diff_cfg)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    decoder_modes = resolve_decoder_modes(args)
    weight_names = ("raw", "ema") if args.weights == "both" else (args.weights,)
    if args.flat_sample_files:
        if not args.per_sample_output_dirs:
            raise ValueError("--flat_sample_files requires --per_sample_output_dirs.")
        if len(decoder_modes) != 1:
            raise ValueError("--flat_sample_files requires exactly one --decoder_mode.")
        if args.weights == "both" or args.debug_latents:
            raise ValueError(
                "--flat_sample_files supports one weight set and does not support debug latents."
            )
    sample_indices = list(
        range(args.sample_index_start, args.sample_index_start + args.n_samples)
    )
    sample_seeds = per_sample_seeds(args.seed, sample_indices)
    checkpoint_mean = sampling_checkpoint.get("latent_mean")
    checkpoint_std = sampling_checkpoint.get("latent_std")
    latent_statistics_hash = (
        canonical_hash(
            {
                "mean": checkpoint_mean.detach().cpu().tolist(),
                "std": checkpoint_std.detach().cpu().tolist(),
            }
        )
        if checkpoint_mean is not None and checkpoint_std is not None
        else None
    )
    resolved_scale_factor = resolve_scale_factor(
        args.scale_factor,
        diff_cfg,
        sampling_checkpoint.get("scale_factor"),
    )

    provenance = {
        "configs": {
            "stage1": file_identity(args.stage1_cfg),
            "diffusion": (
                file_identity(args.diff_cfg)
                if args.diff_cfg is not None
                else {
                    "source": "checkpoint.resolved_config",
                    "sha256": canonical_hash(OmegaConf.to_container(diff_cfg, resolve=True)),
                }
            ),
        },
        "checkpoints": {
            "stage1": file_identity(args.stage1_ckpt),
            "diffusion": file_identity(args.diff_ckpt),
        },
        "preprocessing": {
            "latent_shape": [C, D, H, W, T],
            "requested_image_frames": args.T_image,
            "vqgan_temporal_stride": args.vqgan_temporal_stride,
            "scale_factor": resolved_scale_factor,
            "scale_factor_override": args.scale_factor,
            "normalize_latents": bool(diff_cfg.training.get("normalize_latents", False)),
            "checkpoint_normalize_latents": sampling_checkpoint.get("normalize_latents"),
            "latent_statistics_sha256": latent_statistics_hash,
            "checkpoint_contract": sampling_checkpoint.get("provenance", {}).get(
                "latent_preprocessing"
            ),
            "alpha_t": file_identity(args.alpha_t_path) if args.alpha_t_path else None,
            "geometry": {
                "type": args.geometry,
                "template": file_identity(args.template_nifti) if args.template_nifti else None,
                "canonical_spacing": [float(value) for value in args.spacing]
                if args.geometry == "canonical_synthetic"
                else None,
            },
        },
        "software": software_versions(),
        "frozen_manifest": file_identity(args.frozen_manifest) if args.frozen_manifest else None,
    }
    request = {
        "seed": int(args.seed),
        "per_sample_seeds": sample_seeds,
        "sample_indices": sample_indices,
        "scheduler": {
            "name": scheduler_name,
            "requested_steps": int(args.timesteps),
            "timestep_grid": scheduler.timesteps.detach().cpu().tolist(),
            "solver_order": int(getattr(scheduler, "solver_order", 1)),
        },
        "weights": args.weights,
        "decoder_modes": list(decoder_modes),
        "output_layout": args.output_layout,
        "output_transform": {key: getattr(args, key, default) for key, default in (
            ("output_axes", "dhw"), ("flip_axes", []), ("foreground_crop", False),
            ("foreground_threshold", -0.95), ("foreground_min_fraction", 0.005),
        )},
        **provenance,
    }
    pending_indices = list(sample_indices)
    if args.per_sample_output_dirs:
        if not args.force:
            pending_indices = [
                sample_index
                for sample_index, sample_seed in zip(sample_indices, sample_seeds)
                if not (
                    flat_completion_matches(
                        out_dir,
                        sample_index,
                        _individual_sample_request(request, sample_index, sample_seed),
                    )
                    if args.flat_sample_files
                    else completion_matches(
                        out_dir / f"sample_{sample_index:03d}",
                        _individual_sample_request(request, sample_index, sample_seed),
                    )
                )
            ]
        if not pending_indices:
            print(f"[skip] all per-sample identities and output checksums match {out_dir}")
            return 0
    else:
        if not args.force and completion_matches(out_dir, request):
            print(f"[skip] identity and output checksums match {out_dir}")
            return 0
        (out_dir / "completion_manifest.json").unlink(missing_ok=True)

    seed_by_index = dict(zip(sample_indices, sample_seeds))

    print("Loading VQ-GAN...")
    stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)
    alpha_t = load_alpha_t(args, T, device)

    print(
        f"Sampling {len(pending_indices)}/{args.n_samples} pending volumes "
        f"({scheduler_name.upper()}, {args.timesteps} steps, batch_size={args.batch_size})..."
    )
    print(f"Decoder modes: {', '.join(decoder_modes)}")
    print(f"Weights: {', '.join(weight_names)}")
    print(f"Latent shape: {[C, D, H, W, T]}")
    if alpha_t is not None:
        print(f"Loaded alpha_t descriptor: {args.alpha_t_path} -> {tuple(alpha_t.shape)}")

    saved_paths = []
    sample_saved_paths = {sample_index: [] for sample_index in pending_indices}
    last_metadata_inputs = None
    for weight_index, weights in enumerate(weight_names):
        print(f"Loading DiT4D ({weights})...")
        dit, loaded_cfg, latent_mean, latent_std, ckpt_metadata = load_dit(
            args.diff_cfg,
            args.diff_ckpt,
            device,
            weights=weights,
            checkpoint=sampling_checkpoint,
        )
        validate_normalization_metadata(loaded_cfg, ckpt_metadata)
        if latent_mean is None and diff_cfg.training.get("normalize_latents", False):
            raise RuntimeError(
                "Config has normalize_latents=true but checkpoint lacks latent_mean/latent_std."
            )
        scale_factor = resolve_scale_factor(
            args.scale_factor, diff_cfg, ckpt_metadata.get("scale_factor")
        )
        last_metadata_inputs = (ckpt_metadata, latent_mean, scale_factor)
        print(f"Scale factor ({weights}): {scale_factor}")

        for batch_start in range(0, len(pending_indices), args.batch_size):
            batch_indices = pending_indices[batch_start:batch_start + args.batch_size]
            batch_seeds = [seed_by_index[index] for index in batch_indices]
            if isinstance(scheduler, DPMSolverPPScheduler):
                scheduler.reset_trajectory()
            torch.manual_seed(batch_seeds[0])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(batch_seeds[0])
            generators = [
                torch.Generator(device=device).manual_seed(sample_seed)
                for sample_seed in batch_seeds
            ]
            x = torch.cat(
                [
                    torch.randn(
                        1, C, D, H, W, T, device=device, generator=generator
                    )
                    for generator in generators
                ],
                dim=0,
            )

            x_self_cond = None
            with autocast_context(device, amp_dtype):
                for timestep in scheduler.timesteps:
                    if is_flow_matching:
                        t_embed = timestep.item() * scheduler.num_train_timesteps
                        t_batch = torch.full(
                            (len(batch_indices),),
                            t_embed,
                            device=device,
                            dtype=torch.float32,
                        )
                    else:
                        t_batch = torch.full(
                            (len(batch_indices),), timestep, device=device, dtype=torch.long
                        )
                    noise_pred = dit(
                        x, t=t_batch, y=None, x_self_cond=x_self_cond, alpha_t=alpha_t
                    )
                    x, x0_pred = _scheduler_step_batch(
                        scheduler, noise_pred, timestep, x, generators
                    )
                    if dit.self_conditioning:
                        x_self_cond = x0_pred.detach()

            x = x / scale_factor
            if latent_mean is not None:
                x = x * (latent_std + 1e-8) + latent_mean
            for batch_offset, (sample_index, sample_seed) in enumerate(
                zip(batch_indices, batch_seeds)
            ):
                sample_root = (
                    out_dir
                    if args.flat_sample_files
                    else out_dir / f"sample_{sample_index:03d}"
                    if args.per_sample_output_dirs
                    else out_dir
                )
                weight_root = (
                    sample_root / weights if args.weights == "both" else sample_root
                )
                latent = x[batch_offset].detach()
                if args.debug_latents:
                    latent_path = weight_root / f"sample_{sample_index:03d}_latent.pt"
                    atomic_torch_save(latent.cpu(), latent_path)
                    saved_paths.append(latent_path)
                    sample_saved_paths[sample_index].append(latent_path)

                decoded = decode_latent_modes(stage1, latent, decoder_modes)
                for mode, volume in decoded.items():
                    volume = volume.float().cpu().clamp(-1.0, 1.0).numpy()[0]
                    if args.T_image is not None:
                        if volume.shape[-1] < args.T_image:
                            raise ValueError("Decoded sequence is shorter than requested T_image")
                        volume = volume[..., :args.T_image]
                    mode_root = (
                        weight_root if args.flat_sample_files else weight_root / f"decode_{mode}"
                    )
                    paths = save_output(
                        volume,
                        mode_root,
                        sample_index,
                        args.spacing,
                        args.output_layout,
                        geometry=args.geometry,
                        template_nifti=args.template_nifti,
                        output_axes=args.output_axes,
                        flip_axes=args.flip_axes,
                        foreground_crop=args.foreground_crop,
                        foreground_threshold=args.foreground_threshold,
                        foreground_min_fraction=args.foreground_min_fraction,
                    )
                    saved_paths.extend(paths)
                    sample_saved_paths[sample_index].extend(paths)
                    print(
                        f"  Saved {len(paths)} {mode} file(s) for sample_{sample_index:03d} "
                        f"seed={sample_seed}"
                    )
                if args.per_sample_output_dirs and weight_index == len(weight_names) - 1:
                    sample_metadata = build_sample_metadata(
                        args=args,
                        diff_cfg=diff_cfg,
                        ckpt_metadata=ckpt_metadata,
                        scheduler_name=scheduler_name,
                        scale_factor=scale_factor,
                        latent_shape=[C, D, H, W, T],
                        amp_dtype=amp_dtype,
                        latent_mean=latent_mean,
                        scheduler=scheduler,
                        provenance=provenance,
                        decoder_modes=decoder_modes,
                        sample_seeds=[sample_seed],
                    )
                    sample_metadata["n_samples"] = 1
                    if args.flat_sample_files:
                        metadata_path = (
                            out_dir / ".metadata" / f"sample_{sample_index:03d}.json"
                        )
                        atomic_json_save(_jsonify(sample_metadata), metadata_path)
                    else:
                        metadata_path = write_sample_metadata(sample_root, sample_metadata)
                    saved_paths.append(metadata_path)
                    sample_saved_paths[sample_index].append(metadata_path)
                    individual_request = _individual_sample_request(
                        request, sample_index, sample_seed
                    )
                    if args.flat_sample_files:
                        manifest_path = publish_flat_completion_manifest(
                            out_dir,
                            sample_index,
                            individual_request,
                            sample_saved_paths[sample_index],
                        )
                    else:
                        manifest_path = publish_completion_manifest(
                            sample_root,
                            individual_request,
                            sample_saved_paths[sample_index],
                        )
                    print(f"Wrote completion manifest last: {manifest_path}")
                del latent, decoded
            del x
            torch.cuda.empty_cache()
        del dit

    if not args.per_sample_output_dirs:
        ckpt_metadata, latent_mean, scale_factor = last_metadata_inputs
        metadata = build_sample_metadata(
            args=args,
            diff_cfg=diff_cfg,
            ckpt_metadata=ckpt_metadata,
            scheduler_name=scheduler_name,
            scale_factor=scale_factor,
            latent_shape=[C, D, H, W, T],
            amp_dtype=amp_dtype,
            latent_mean=latent_mean,
            scheduler=scheduler,
            provenance=provenance,
            decoder_modes=decoder_modes,
            sample_seeds=sample_seeds,
        )
        metadata_path = write_sample_metadata(out_dir, metadata)
        saved_paths.append(metadata_path)
        manifest_path = publish_completion_manifest(out_dir, request, saved_paths)
        print(f"Wrote completion manifest last: {manifest_path}")
    print(f"\nDone. {args.n_samples} samples saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

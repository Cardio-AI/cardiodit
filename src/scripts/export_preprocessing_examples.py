"""
Export image-space preprocessing examples for temporal-alignment experiments.

For each selected CMR, this script saves:
- a spatially preprocessed native-T NIfTI reference,
- Exp 1 cyclic, Exp 2 linear, Exp 3 piecewise, and Exp 4 Fourier T=32 NIfTIs,
- optional Exp 5 descriptor-DTW preview NIfTI when descriptor sidecars are provided,
- original-vs-output keyframe difference-map NIfTIs,
- side-by-side and pairwise difference GIFs over one z-slice.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping
import json
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import RUNS_ROOT

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.temporal_align import (
    DEFAULT_PIECEWISE_TARGETS,
    PIECEWISE_PHASE_ORDER,
    align,
)
from src.utils.descriptor_io import (
    TEMPLATE_KEYS,
    load_1d_tensor,
    load_descriptor_sidecar,
)


METHODS = {
    "exp1_cyclic": "cyclic",
    "exp2_linear": "linear",
    "exp3_piecewise": "piecewise",
    "exp4_fourier": "fourier",
}


@dataclass(frozen=True)
class PreprocessedNative:
    data: np.ndarray
    affine: np.ndarray
    source_affine: np.ndarray
    source_shape: tuple[int, ...]
    crop_pad_offsets_hwd: tuple[int, int, int]
    time_zooms: tuple[float, ...]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export image-space NIfTI and GIF examples for preprocessing experiments."
    )
    parser.add_argument("--csv", required=True, help="CSV with an image column.")
    parser.add_argument("--output_dir", default=str(RUNS_ROOT / "temporal_alignment" / "preprocessing_examples"))
    parser.add_argument("--keyframe_dir", required=True)
    parser.add_argument("--descriptor_sidecar_dir", default=None,
                        help="Optional sidecar root for descriptor-DTW preview. "
                             "Looks for .pt or .npy descriptors under alpha_t/, "
                             "descriptors/, or motion_descriptor/.")
    parser.add_argument("--dtw_template", default=None,
                        help="Optional 1D .pt or .npy DTW template. If omitted with "
                             "--descriptor_sidecar_dir, a toy template is made by "
                             "linear-resampling the subject descriptor to "
                             "--target_frames.")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Optional subject stems, e.g. 002_SA_CINE 003_SA_CINE.")
    parser.add_argument("--target_frames", type=int, default=32)
    parser.add_argument("--roi_size", type=int, nargs=2, default=[256, 256],
                        metavar=("H", "W"))
    parser.add_argument("--target_z", type=int, default=12)
    parser.add_argument("--slice_index", type=int, default=None,
                        help="Z slice for GIFs. Defaults to the center slice after preprocessing.")
    parser.add_argument("--gif_fps", type=float, default=8.0)
    parser.add_argument("--diff_percentile", type=float, default=99.0,
                        help="Percentile used to scale absolute/signed diff GIF panels.")
    return parser.parse_args()


def subject_id(path: str | Path) -> str:
    return Path(path).stem.replace(".nii", "")


def select_image_paths(csv_path: Path, subjects: list[str] | None, num_examples: int) -> list[Path]:
    with open(csv_path, newline="") as f:
        rows = [row["image"] for row in csv.DictReader(f)]

    if subjects:
        wanted = set(subjects)
        selected = [Path(p) for p in rows if subject_id(p) in wanted]
        missing = sorted(wanted - {subject_id(p) for p in selected})
        if missing:
            raise FileNotFoundError(f"Subjects not found in {csv_path}: {missing}")
        return selected

    return [Path(p) for p in rows[:num_examples]]


def scale_intensity(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.full_like(x, -1.0, dtype=np.float32)
    return ((x - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def crop_or_pad_axis(x: np.ndarray, axis: int, target: int, pad_value: float) -> tuple[np.ndarray, int]:
    size = x.shape[axis]
    offset = 0
    if size > target:
        start = (size - target) // 2
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(start, start + target)
        x = x[tuple(slices)]
        size = target
        offset = start
    if size < target:
        before = (target - size) // 2
        after = target - size - before
        pads = [(0, 0)] * x.ndim
        pads[axis] = (before, after)
        x = np.pad(x, pads, mode="constant", constant_values=pad_value)
        offset = -before
    return x, offset


def source_spatial_affine(img: nib.Nifti1Image) -> np.ndarray:
    header = img.header
    if int(header["qform_code"]) > 0:
        return np.asarray(img.get_qform(), dtype=np.float64)
    if int(header["sform_code"]) > 0:
        return np.asarray(img.get_sform(), dtype=np.float64)
    return np.asarray(img.affine, dtype=np.float64)


def internal_grid_affine(source_affine: np.ndarray, offsets_hwd: tuple[int, int, int]) -> np.ndarray:
    offset_h, offset_w, offset_d = offsets_hwd
    output_affine = np.eye(4, dtype=np.float64)
    # Source MNM2 axes are (Z, H, W); internal/export axes are (H, W, Z).
    output_affine[:3, 0] = source_affine[:3, 1]
    output_affine[:3, 1] = source_affine[:3, 2]
    output_affine[:3, 2] = source_affine[:3, 0]
    source_offset_zhw = np.asarray([offset_d, offset_h, offset_w], dtype=np.float64)
    output_affine[:3, 3] = source_affine[:3, 3] + source_affine[:3, :3] @ source_offset_zhw
    return output_affine


def load_preprocessed_native(path: Path, roi_size: tuple[int, int], target_z: int) -> PreprocessedNative:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"{path}: expected 4D NIfTI, got shape {data.shape}")

    # MNM2 files are laid out as (Z, H, W, T). Encoding uses (H, W, D, T).
    data = np.transpose(data, (1, 2, 0, 3))
    data = scale_intensity(data)

    target_h, target_w = roi_size
    data, offset_h = crop_or_pad_axis(data, axis=0, target=target_h, pad_value=-1.0)
    data, offset_w = crop_or_pad_axis(data, axis=1, target=target_w, pad_value=-1.0)
    data, offset_d = crop_or_pad_axis(data, axis=2, target=target_z, pad_value=-1.0)

    source_affine = source_spatial_affine(img)
    offsets_hwd = (offset_h, offset_w, offset_d)
    return PreprocessedNative(
        data=data.astype(np.float32, copy=False),
        affine=internal_grid_affine(source_affine, offsets_hwd),
        source_affine=source_affine,
        source_shape=tuple(int(v) for v in img.shape),
        crop_pad_offsets_hwd=offsets_hwd,
        time_zooms=tuple(float(v) for v in img.header.get_zooms()[3:]),
    )


def load_keyframes(keyframe_dir: Path, stem: str) -> dict[str, int]:
    path = keyframe_dir / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing keyframe file: {path}")
    with open(path) as f:
        return json.load(f)


def apply_method(
    native: np.ndarray,
    method: str,
    target_frames: int,
    keyframes: dict[str, int] | None,
    descriptor: torch.Tensor | None = None,
    template: torch.Tensor | None = None,
) -> np.ndarray:
    x = torch.from_numpy(native).unsqueeze(0)
    kwargs = {"keyframes": keyframes} if method == "piecewise" else {}
    if method == "dtw":
        if descriptor is None or template is None:
            raise ValueError("DTW export requires descriptor and template tensors")
        kwargs = {"descriptor": descriptor, "template": template}
    y = align(x, method=method, T_out=target_frames, **kwargs)
    return y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def make_toy_dtw_template(descriptor: torch.Tensor, target_frames: int) -> torch.Tensor:
    x = descriptor.detach().cpu().float().view(1, 1, -1)
    return F.interpolate(x, size=target_frames, mode="linear", align_corners=False).view(-1)


def load_dtw_export_inputs(
    descriptor_sidecar_dir: Path | None,
    template_path: Path | None,
    stem: str,
    target_frames: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
    if descriptor_sidecar_dir is None:
        return None, None, {}

    descriptor, descriptor_path = load_descriptor_sidecar(
        descriptor_sidecar_dir,
        stem,
        "DTW descriptor",
    )
    if template_path is None:
        template = make_toy_dtw_template(descriptor, target_frames)
        template_source = "toy_linear_resample_of_subject_descriptor"
    else:
        template = load_1d_tensor(template_path, "DTW template", keys=TEMPLATE_KEYS)
        template_source = str(template_path)

    if template.numel() != target_frames:
        raise ValueError(
            f"DTW template length {template.numel()} != --target_frames={target_frames}"
        )

    return descriptor, template, {
        "descriptor_path": str(descriptor_path),
        "template_source": template_source,
        "descriptor_length": int(descriptor.numel()),
        "template_length": int(template.numel()),
    }


def save_nifti(
    path: Path,
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    time_zoom: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(volume.astype(np.float32, copy=False), affine=affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_data_dtype(np.float32)
    image.header.set_xyzt_units("mm", "sec")
    if time_zoom is not None and volume.ndim >= 4:
        image.header["pixdim"][4] = float(time_zoom)
    nib.save(image, path)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, -1.0, 1.0)
    return ((frame + 1.0) * 127.5).astype(np.uint8)


def make_foreground_mask(native: np.ndarray) -> np.ndarray:
    mask = np.max(native, axis=-1) > -0.95
    if not np.any(mask):
        return np.ones(native.shape[:-1], dtype=bool)
    return mask


def labeled_panel(frame: np.ndarray, label: str, width: int, label_h: int = 28) -> Image.Image:
    img = Image.fromarray(to_uint8(frame)).convert("RGB")
    if img.width != width:
        new_h = max(1, round(img.height * width / img.width))
        img = img.resize((width, new_h), resample=Image.Resampling.BILINEAR)

    panel = Image.new("RGB", (img.width, img.height + label_h), color=(18, 18, 18))
    panel.paste(img, (0, label_h))

    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 8), label, fill=(240, 240, 240), font=font)
    return panel


def rgb_panel(rgb: np.ndarray, label: str, width: int, label_h: int = 28) -> Image.Image:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGB")
    if img.width != width:
        new_h = max(1, round(img.height * width / img.width))
        img = img.resize((width, new_h), resample=Image.Resampling.BILINEAR)

    panel = Image.new("RGB", (img.width, img.height + label_h), color=(18, 18, 18))
    panel.paste(img, (0, label_h))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), label, fill=(240, 240, 240), font=ImageFont.load_default())
    return panel


def absdiff_rgb(diff: np.ndarray, vmax: float) -> np.ndarray:
    scaled = np.clip(np.abs(diff) / max(vmax, 1e-8), 0.0, 1.0)
    rgb = np.zeros((*diff.shape, 3), dtype=np.float32)
    rgb[..., 0] = scaled * 255.0
    rgb[..., 1] = scaled * 210.0
    return rgb.astype(np.uint8)


def signed_diff_rgb(diff: np.ndarray, vmax: float) -> np.ndarray:
    scaled = np.clip(diff / max(vmax, 1e-8), -1.0, 1.0)
    rgb = np.full((*diff.shape, 3), 30.0, dtype=np.float32)
    pos = scaled > 0
    neg = scaled < 0
    rgb[..., 0] += pos * scaled * 225.0
    rgb[..., 1] += neg * (-scaled) * 170.0
    rgb[..., 2] += neg * (-scaled) * 225.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def absdiff_legend_bar(width: int, height: int) -> Image.Image:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = x * 255.0
    rgb[..., 1] = x * 210.0
    return Image.fromarray(rgb.astype(np.uint8))


def signed_diff_legend_bar(width: int, height: int) -> Image.Image:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    rgb = np.full((height, width, 3), 30.0, dtype=np.float32)
    pos = x > 0
    neg = x < 0
    rgb[..., 0] += pos.reshape(1, -1) * x.clip(min=0).reshape(1, -1) * 225.0
    rgb[..., 1] += neg.reshape(1, -1) * (-x.clip(max=0)).reshape(1, -1) * 170.0
    rgb[..., 2] += neg.reshape(1, -1) * (-x.clip(max=0)).reshape(1, -1) * 225.0
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def draw_diff_legend(
    canvas: Image.Image,
    y0: int,
    left_label: str,
    right_label: str,
    vmax: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text = (
        f"Difference convention: right - left = {right_label} - {left_label}. "
        "Abs diff: black->yellow means larger |difference|. "
        "Signed diff: cyan means right lower, red means right higher, gray means near zero."
    )
    draw.text((10, y0 + 6), text, fill=(235, 235, 235), font=font)

    bar_w = min(260, max(120, canvas.width // 4))
    bar_h = 12
    abs_x = 10
    signed_x = abs_x + bar_w + 230
    bar_y = y0 + 32

    canvas.paste(absdiff_legend_bar(bar_w, bar_h), (abs_x, bar_y))
    draw.text((abs_x + bar_w + 8, bar_y - 1), f"abs: 0 -> {vmax:.4f}", fill=(235, 235, 235), font=font)

    canvas.paste(signed_diff_legend_bar(bar_w, bar_h), (signed_x, bar_y))
    draw.text(
        (signed_x + bar_w + 8, bar_y - 1),
        f"signed: -{vmax:.4f} -> +{vmax:.4f}",
        fill=(235, 235, 235),
        font=font,
    )


def make_comparison_gif(
    path: Path,
    native: np.ndarray,
    outputs: dict[str, np.ndarray],
    method_names: Mapping[str, str],
    z_idx: int,
    target_frames: int,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000.0 / fps
    panel_w = native.shape[0]
    frames = []

    labels = [("native", native)] + [
        (name.replace("_", " "), outputs[name])
        for name in method_names
    ]
    for t in range(target_frames):
        panels = []
        native_t = min(round(t * native.shape[-1] / target_frames), native.shape[-1] - 1)
        for label, vol in labels:
            idx = native_t if label == "native" else t
            frame = vol[:, :, z_idx, idx]
            panels.append(labeled_panel(frame, f"{label}  t={idx:02d}", width=panel_w))
        canvas = Image.new(
            "RGB",
            (sum(p.width for p in panels), max(p.height for p in panels)),
            color=(0, 0, 0),
        )
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        frames.append(canvas)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def make_diff_gif(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    left_label: str,
    right_label: str,
    z_idx: int,
    fps: float,
    percentile: float,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000.0 / fps
    panel_w = left.shape[0]
    diff = right - left
    vmax = float(np.percentile(np.abs(diff[:, :, z_idx, :]), percentile))
    vmax = max(vmax, 1e-6)
    frames = []

    for t in range(left.shape[-1]):
        left_frame = left[:, :, z_idx, t]
        right_frame = right[:, :, z_idx, t]
        diff_frame = diff[:, :, z_idx, t]
        panels = [
            labeled_panel(left_frame, f"{left_label}  t={t:02d}", width=panel_w),
            labeled_panel(right_frame, f"{right_label}  t={t:02d}", width=panel_w),
            rgb_panel(absdiff_rgb(diff_frame, vmax), "abs diff", width=panel_w),
            rgb_panel(signed_diff_rgb(diff_frame, vmax), "signed diff", width=panel_w),
        ]
        legend_h = 58
        panel_h = max(p.height for p in panels)
        canvas = Image.new(
            "RGB",
            (sum(p.width for p in panels), panel_h + legend_h),
            color=(0, 0, 0),
        )
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        draw_diff_legend(canvas, panel_h, left_label, right_label, vmax)
        frames.append(canvas)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return vmax


def masked_values(volume: np.ndarray, mask: np.ndarray, t: int | None = None) -> np.ndarray:
    if t is None:
        return volume[mask, :]
    return volume[..., t][mask]


def pairwise_metric_rows(
    subject: str,
    outputs: dict[str, np.ndarray],
    mask: np.ndarray,
    pairwise_comparisons: list[tuple[str, str]],
) -> tuple[list[dict], list[dict]]:
    frame_rows = []
    summary_rows = []
    for left_name, right_name in pairwise_comparisons:
        diff = outputs[right_name] - outputs[left_name]
        diff_fg = masked_values(diff, mask)
        summary_rows.append({
            "subject": subject,
            "left": left_name,
            "right": right_name,
            "mae_fg": float(np.mean(np.abs(diff_fg))),
            "rmse_fg": float(np.sqrt(np.mean(diff_fg ** 2))),
            "max_abs_fg": float(np.max(np.abs(diff_fg))),
            "mae_all": float(np.mean(np.abs(diff))),
            "rmse_all": float(np.sqrt(np.mean(diff ** 2))),
        })
        for t in range(diff.shape[-1]):
            frame_diff = masked_values(diff, mask, t=t)
            frame_rows.append({
                "subject": subject,
                "left": left_name,
                "right": right_name,
                "frame": t,
                "mae_fg": float(np.mean(np.abs(frame_diff))),
                "rmse_fg": float(np.sqrt(np.mean(frame_diff ** 2))),
                "max_abs_fg": float(np.max(np.abs(frame_diff))),
                "mae_all": float(np.mean(np.abs(diff[..., t]))),
            })
    return frame_rows, summary_rows


def temporal_metric_rows(subject: str, outputs: dict[str, np.ndarray], mask: np.ndarray) -> list[dict]:
    rows = []
    for name, volume in outputs.items():
        delta = np.diff(volume, axis=-1)
        accel = np.diff(volume, n=2, axis=-1)
        for t in range(delta.shape[-1]):
            delta_fg = masked_values(delta, mask, t=t)
            rows.append({
                "subject": subject,
                "method": name,
                "transition": t,
                "mean_abs_delta_fg": float(np.mean(np.abs(delta_fg))),
                "rmse_delta_fg": float(np.sqrt(np.mean(delta_fg ** 2))),
                "mean_abs_delta_all": float(np.mean(np.abs(delta[..., t]))),
                "mean_abs_accel_fg": (
                    float(np.mean(np.abs(masked_values(accel, mask, t=t))))
                    if t < accel.shape[-1] else ""
                ),
            })
    return rows


def cyclic_distance(a: float, b: float, period: int) -> float:
    d = abs(a - b) % period
    return float(min(d, period - d))


def linear_source_position(frame: int, T_in: int, T_out: int) -> float:
    # Matches torch.nn.functional.interpolate(..., align_corners=False).
    return (frame + 0.5) * T_in / T_out - 0.5


def fourier_source_position(frame: int, T_in: int, T_out: int) -> float:
    return frame * T_in / T_out


def keyframe_alignment_rows(
    subject: str,
    keyframes: dict[str, int],
    T_in: int,
    T_out: int,
) -> list[dict]:
    rows = []
    for phase in PIECEWISE_PHASE_ORDER:
        target = int(round(DEFAULT_PIECEWISE_TARGETS[phase] * T_out))
        keyframe = int(keyframes[phase])
        sources = {
            "exp1_cyclic": float(target % T_in),
            "exp2_linear": linear_source_position(target, T_in, T_out),
            "exp3_piecewise": float(keyframe),
            "exp4_fourier": fourier_source_position(target, T_in, T_out),
        }
        for method, source_pos in sources.items():
            rows.append({
                "subject": subject,
                "phase": phase,
                "target_frame": target,
                "method": method,
                "source_position": source_pos,
                "native_keyframe": keyframe,
                "circular_abs_error_frames": cyclic_distance(source_pos, keyframe, T_in),
            })
    return rows


def keyframe_target_frame(phase: str, T_out: int) -> int:
    return min(int(round(DEFAULT_PIECEWISE_TARGETS[phase] * T_out)), T_out - 1)


def save_keyframe_difference_maps(
    subject_dir: Path,
    subject: str,
    native: np.ndarray,
    outputs: dict[str, np.ndarray],
    method_names: Mapping[str, str],
    keyframes: dict[str, int],
    T_out: int,
    affine: np.ndarray,
) -> list[dict]:
    diff_dir = subject_dir / "keyframe_differences"
    diff_dir.mkdir(parents=True, exist_ok=True)

    phases = list(PIECEWISE_PHASE_ORDER)
    native_frames = [int(keyframes[phase]) for phase in phases]
    target_frames = [keyframe_target_frame(phase, T_out) for phase in phases]
    original_stack = np.stack(
        [native[..., frame] for frame in native_frames],
        axis=-1,
    ).astype(np.float32, copy=False)
    save_nifti(diff_dir / "original_keyframes.nii.gz", original_stack, affine, time_zoom=1.0)

    rows = []
    for method_name in method_names:
        output_stack = np.stack(
            [outputs[method_name][..., frame] for frame in target_frames],
            axis=-1,
        ).astype(np.float32, copy=False)
        signed_diff = output_stack - original_stack
        abs_diff = np.abs(signed_diff).astype(np.float32, copy=False)

        save_nifti(diff_dir / f"{method_name}_keyframes.nii.gz", output_stack, affine, time_zoom=1.0)
        save_nifti(diff_dir / f"{method_name}_signed_diff_keyframes.nii.gz", signed_diff, affine, time_zoom=1.0)
        save_nifti(diff_dir / f"{method_name}_abs_diff_keyframes.nii.gz", abs_diff, affine, time_zoom=1.0)

        for phase_idx, phase in enumerate(phases):
            diff = signed_diff[..., phase_idx]
            rows.append({
                "subject": subject,
                "method": method_name,
                "phase": phase,
                "native_frame": native_frames[phase_idx],
                "target_frame": target_frames[phase_idx],
                "mae_all": float(np.mean(np.abs(diff))),
                "rmse_all": float(np.sqrt(np.mean(diff ** 2))),
                "max_abs_all": float(np.max(np.abs(diff))),
            })

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_subject_metrics(
    path: Path,
    subject: str,
    pairwise_rows: list[dict],
    temporal_rows: list[dict],
    method_names: Mapping[str, str],
    pairwise_comparisons: list[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    for left_name, right_name in pairwise_comparisons:
        xs = [r["frame"] for r in pairwise_rows if r["left"] == left_name and r["right"] == right_name]
        ys = [r["mae_fg"] for r in pairwise_rows if r["left"] == left_name and r["right"] == right_name]
        axes[0].plot(xs, ys, label=f"{left_name} vs {right_name}")
    axes[0].set_title(f"{subject}: foreground pairwise MAE per output frame")
    axes[0].set_xlabel("output frame")
    axes[0].set_ylabel("MAE in scaled intensity")
    axes[0].legend(fontsize=8)

    for method in method_names:
        xs = [r["transition"] for r in temporal_rows if r["method"] == method]
        ys = [r["mean_abs_delta_fg"] for r in temporal_rows if r["method"] == method]
        axes[1].plot(xs, ys, label=method)
    axes[1].set_title("Foreground frame-to-frame change")
    axes[1].set_xlabel("transition t -> t+1")
    axes[1].set_ylabel("mean |delta|")
    axes[1].legend(fontsize=8)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    keyframe_dir = Path(args.keyframe_dir)
    descriptor_sidecar_dir = Path(args.descriptor_sidecar_dir) if args.descriptor_sidecar_dir else None
    dtw_template_path = Path(args.dtw_template) if args.dtw_template else None
    image_paths = select_image_paths(Path(args.csv), args.subjects, args.num_examples)
    roi_size = (int(args.roi_size[0]), int(args.roi_size[1]))
    method_names = dict(METHODS)
    if descriptor_sidecar_dir is not None:
        method_names["exp5_dtw_toy" if dtw_template_path is None else "exp5_dtw"] = "dtw"
    pairwise_comparisons = list(combinations(method_names.keys(), 2))

    manifest = {
        "csv": args.csv,
        "keyframe_dir": str(keyframe_dir),
        "descriptor_sidecar_dir": str(descriptor_sidecar_dir) if descriptor_sidecar_dir else None,
        "dtw_template": str(dtw_template_path) if dtw_template_path else None,
        "target_frames": args.target_frames,
        "roi_size": list(roi_size),
        "target_z": args.target_z,
        "piecewise_phase_order": list(PIECEWISE_PHASE_ORDER),
        "piecewise_default_targets": DEFAULT_PIECEWISE_TARGETS,
        "affine_policy": "source_qform_preferred_then_sform_after_H_W_Z_transpose_and_center_crop_pad",
        "keyframe_difference_stack_order": list(PIECEWISE_PHASE_ORDER),
        "methods": method_names,
        "pairwise_comparisons": pairwise_comparisons,
        "subjects": [],
    }

    all_pairwise = []
    all_pairwise_summary = []
    all_temporal = []
    all_keyframe_alignment = []
    all_keyframe_differences = []

    for image_path in image_paths:
        stem = subject_id(image_path)
        subject_dir = out_root / stem
        native_info = load_preprocessed_native(image_path, roi_size=roi_size, target_z=args.target_z)
        native = native_info.data
        keyframes = load_keyframes(keyframe_dir, stem)
        descriptor, template, dtw_metadata = load_dtw_export_inputs(
            descriptor_sidecar_dir,
            dtw_template_path,
            stem,
            args.target_frames,
        )

        outputs = {
            name: apply_method(
                native,
                method=method,
                target_frames=args.target_frames,
                keyframes=keyframes if method == "piecewise" else None,
                descriptor=descriptor if method == "dtw" else None,
                template=template if method == "dtw" else None,
            )
            for name, method in method_names.items()
        }
        mask = make_foreground_mask(native)

        z_idx = args.slice_index if args.slice_index is not None else native.shape[2] // 2
        if z_idx < 0 or z_idx >= native.shape[2]:
            raise ValueError(f"--slice_index {z_idx} outside [0, {native.shape[2]})")

        source_time_zoom = native_info.time_zooms[0] if native_info.time_zooms else 1.0
        save_nifti(
            subject_dir / "native_preprocessed.nii.gz",
            native,
            native_info.affine,
            time_zoom=source_time_zoom,
        )
        for name, volume in outputs.items():
            save_nifti(
                subject_dir / f"{name}.nii.gz",
                volume,
                native_info.affine,
                time_zoom=source_time_zoom,
            )

        gif_path = subject_dir / f"comparison_z{z_idx:02d}.gif"
        make_comparison_gif(
            gif_path,
            native=native,
            outputs=outputs,
            method_names=method_names,
            z_idx=z_idx,
            target_frames=args.target_frames,
            fps=args.gif_fps,
        )

        diff_scales = {}
        for left_name, right_name in pairwise_comparisons:
            diff_path = subject_dir / f"diff_{left_name}_vs_{right_name}_z{z_idx:02d}.gif"
            diff_scales[f"{left_name}_vs_{right_name}"] = make_diff_gif(
                diff_path,
                outputs[left_name],
                outputs[right_name],
                left_label=left_name,
                right_label=right_name,
                z_idx=z_idx,
                fps=args.gif_fps,
                percentile=args.diff_percentile,
            )

        pairwise_rows, pairwise_summary_rows = pairwise_metric_rows(
            stem,
            outputs,
            mask,
            pairwise_comparisons,
        )
        temporal_rows = temporal_metric_rows(stem, outputs, mask)
        alignment_rows = keyframe_alignment_rows(
            stem,
            keyframes=keyframes,
            T_in=native.shape[-1],
            T_out=args.target_frames,
        )
        keyframe_diff_rows = save_keyframe_difference_maps(
            subject_dir,
            stem,
            native,
            outputs,
            method_names,
            keyframes,
            args.target_frames,
            native_info.affine,
        )
        write_csv(subject_dir / "pairwise_metrics_by_frame.csv", pairwise_rows)
        write_csv(subject_dir / "pairwise_metrics_summary.csv", pairwise_summary_rows)
        write_csv(subject_dir / "temporal_metrics_by_transition.csv", temporal_rows)
        write_csv(subject_dir / "keyframe_alignment.csv", alignment_rows)
        write_csv(subject_dir / "keyframe_differences.csv", keyframe_diff_rows)
        plot_subject_metrics(
            subject_dir / "metrics_summary.png",
            stem,
            pairwise_rows,
            temporal_rows,
            method_names,
            pairwise_comparisons,
        )

        all_pairwise.extend(pairwise_rows)
        all_pairwise_summary.extend(pairwise_summary_rows)
        all_temporal.extend(temporal_rows)
        all_keyframe_alignment.extend(alignment_rows)
        all_keyframe_differences.extend(keyframe_diff_rows)

        with open(subject_dir / "metadata.json", "w") as f:
            json.dump({
                "image": str(image_path),
                "subject": stem,
                "source_shape": list(native_info.source_shape),
                "native_shape": list(native.shape),
                "output_shape": list(next(iter(outputs.values())).shape),
                "source_affine": native_info.source_affine.tolist(),
                "output_affine": native_info.affine.tolist(),
                "crop_pad_offsets_hwd": list(native_info.crop_pad_offsets_hwd),
                "keyframes": keyframes,
                "dtw": dtw_metadata,
                "slice_index": z_idx,
                "gif": str(gif_path),
                "foreground_voxels": int(np.sum(mask)),
                "diff_scales": diff_scales,
            }, f, indent=2)

        manifest["subjects"].append(stem)
        print(f"Saved {subject_dir}")

    write_csv(out_root / "pairwise_metrics_by_frame.csv", all_pairwise)
    write_csv(out_root / "pairwise_metrics_summary.csv", all_pairwise_summary)
    write_csv(out_root / "temporal_metrics_by_transition.csv", all_temporal)
    write_csv(out_root / "keyframe_alignment.csv", all_keyframe_alignment)
    write_csv(out_root / "keyframe_differences.csv", all_keyframe_differences)

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

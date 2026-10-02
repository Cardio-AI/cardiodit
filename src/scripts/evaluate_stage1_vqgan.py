"""
CPU-only reconstruction and latent-space evaluation for trained stage1 VQ-GANs.

The evaluator reconstructs every z-slice of each held-out 4D CMR volume after
applying the same deterministic preprocessing used for stage1 training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()
sys.path.append(str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
from skimage.metrics import structural_similarity
from sklearn.manifold import TSNE

from src.data.dataloading import CyclicPadTimed, PadTimeToMultipleD, PermuteDimensionsd
from src.models.vqvae import VQVAE
from src.utils.sample_integrity import file_identity, resolve_checkpoint_reference
from src.utils.stage1_loading import load_stage1_strict


RUN_NAMES = (
    "S1_ds4_all_dims_fixed32",
    "S1_ds4_all_dims_paddiv",
    "S1_ds4xy_noT_native",
    "S1_ds8xy_noT_native",
)

DISPLAY_NAMES = {
    "S1_ds4_all_dims_fixed32": "ds4 fixed32",
    "S1_ds4_all_dims_paddiv": "ds4 paddiv",
    "S1_ds4xy_noT_native": "ds4xy noT",
    "S1_ds8xy_noT_native": "ds8xy noT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained stage1 VQ-GAN checkpoints on CPU.")
    parser.add_argument("--csv", required=True, help="Independent test-split manifest.")
    parser.add_argument("--split_name", default="test")
    parser.add_argument("--allow_non_test_split", action="store_true")
    parser.add_argument("--run", action="append", choices=RUN_NAMES, help="Run name. Repeat to evaluate multiple.")
    parser.add_argument("--config_dir", default=str(REPO_ROOT / "configs" / "stage1"))
    parser.add_argument("--checkpoint_root", default=str(RUNS_ROOT / "outputs" / "stage1"))
    parser.add_argument("--checkpoint_name", default="best_model.pth")
    parser.add_argument("--output_dir", default=str(RUNS_ROOT / "outputs" / "stage1_eval"))
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Number of z-slices decoded at once.")
    parser.add_argument("--slice_mode", choices=("all", "center", "linspace"), default="all")
    parser.add_argument("--slice_count", type=int, default=1, help="Used with --slice_mode linspace.")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--latent_value_samples", type=int, default=200_000)
    parser.add_argument("--latent_vector_samples", type=int, default=5_000)
    parser.add_argument("--tokens_per_subject", type=int, default=128)
    parser.add_argument("--plots_only", action="store_true", help="Regenerate plots from existing eval artifacts.")
    parser.add_argument("--input_bits_per_voxel", type=int, default=16)
    parser.add_argument("--downstream_metrics_json", default=None)
    parser.add_argument("--allow_missing_downstream", action="store_true")
    parser.add_argument("--diagnostic_subset", action="store_true")
    return parser.parse_args()


def resolve_image_path(path_value: str, manifest_dir: Path | None = None) -> str:
    path = Path(os.path.expandvars(path_value)).expanduser()
    if not path.is_absolute() and manifest_dir is not None:
        path = manifest_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Manifest image does not exist: {path}")
    return str(path.resolve())


def load_rows(csv_path: Path, max_subjects: int | None) -> list[dict[str, str]]:
    rows = pd.read_csv(csv_path).to_dict("records")
    if max_subjects is not None:
        rows = rows[: max(0, max_subjects)]
    resolved = []
    for row in rows:
        item = dict(row)
        item["image"] = resolve_image_path(str(row["image"]), csv_path.resolve().parent)
        resolved.append(item)
    return resolved


def parse_target_frames(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "native":
        return None
    return int(value)


def build_eval_transform(config) -> Compose:
    roi_size = tuple(int(v) for v in config.training.roi_size)
    target_frames = parse_target_frames(config.training.get("target_frames", roi_size[-1]))
    time_pad_multiple = config.training.get("time_pad_multiple", None)
    if target_frames is not None and time_pad_multiple is not None:
        raise ValueError("Config sets both target_frames and time_pad_multiple.")

    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]
    spatial_permute = config.training.get("spatial_permute", None)
    if spatial_permute is not None:
        transforms.append(PermuteDimensionsd(keys=["image"], perm=tuple(int(v) for v in spatial_permute)))

    transforms.append(ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0))
    if target_frames is not None:
        transforms.append(CyclicPadTimed(keys=["image"], target_frames=target_frames, dim=0))
    elif time_pad_multiple is not None:
        transforms.append(PadTimeToMultipleD(keys=["image"], multiple=int(time_pad_multiple), dim=0))

    roi_xy = (roi_size[0], roi_size[1], -1)
    transforms.extend(
        [
            CenterSpatialCropd(keys=["image"], roi_size=roi_xy),
            SpatialPadd(keys=["image"], spatial_size=roi_xy, constant_values=-1.0),
            ToTensord(keys=["image"]),
        ]
    )
    return Compose(transforms)


def load_model(config_path: Path, checkpoint_path: Path) -> tuple[VQVAE, dict, dict]:
    checkpoint_path = resolve_checkpoint_reference(checkpoint_path)
    model, _, identity = load_stage1_strict(config_path, checkpoint_path, "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    return model, ckpt, identity


def mse_to_psnr(mse: float, data_range: float = 2.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def mean_frame_ssim(input_thwd: np.ndarray, recon_thwd: np.ndarray) -> tuple[float, int]:
    values = []
    t_count, _, _, d_count = input_thwd.shape
    for d_idx in range(d_count):
        for t_idx in range(t_count):
            values.append(
                structural_similarity(
                    input_thwd[t_idx, :, :, d_idx],
                    recon_thwd[t_idx, :, :, d_idx],
                    data_range=2.0,
                )
            )
    return float(np.mean(values)), len(values)


def masked_reconstruction_metrics(
    input_thwd: torch.Tensor,
    recon_thwd: torch.Tensor,
    valid_frames: torch.Tensor,
) -> dict[str, float | int]:
    valid_frames = torch.as_tensor(valid_frames, dtype=torch.bool)
    if valid_frames.ndim != 1 or valid_frames.numel() != input_thwd.shape[0]:
        raise ValueError("valid_frames must have one entry per temporal frame.")
    if not bool(valid_frames.any()):
        raise ValueError("At least one temporal frame must be valid.")
    source = input_thwd[valid_frames]
    reconstruction = recon_thwd[valid_frames]
    diff = (reconstruction - source).float()
    sse = float(diff.square().sum().item())
    absolute = float(diff.abs().sum().item())
    voxels = int(diff.numel())
    ssim, ssim_count = mean_frame_ssim(source.numpy(), reconstruction.numpy())
    return {
        "sse": sse,
        "absolute_error": absolute,
        "voxels": voxels,
        "mse": sse / voxels,
        "mae": absolute / voxels,
        "ssim": ssim,
        "ssim_count": ssim_count,
        "valid_frames": int(valid_frames.sum()),
    }


def rate_metrics(token_count: int, num_embeddings: int, valid_voxels: int, input_bits: int = 16) -> dict:
    bits_per_token = int(math.ceil(math.log2(max(int(num_embeddings), 2))))
    encoded_bits = int(token_count) * bits_per_token
    bits_per_voxel = encoded_bits / max(int(valid_voxels), 1)
    return {
        "latent_tokens": int(token_count),
        "bits_per_token": bits_per_token,
        "theoretical_encoded_bits": encoded_bits,
        "bits_per_valid_voxel": bits_per_voxel,
        "theoretical_compression_ratio": float(input_bits) / max(bits_per_voxel, 1e-12),
    }


def project_temporal_validity(valid_frames: torch.Tensor, target_length: int) -> torch.Tensor:
    """Project image-frame validity to latent token centers."""
    valid_frames = torch.as_tensor(valid_frames, dtype=torch.bool)
    if valid_frames.ndim != 1 or valid_frames.numel() < 1 or target_length < 1:
        raise ValueError("Temporal validity and target length must be non-empty.")
    positions = (
        (torch.arange(target_length, dtype=torch.float64) + 0.5)
        * valid_frames.numel()
        / target_length
        - 0.5
    ).round().long().clamp(0, valid_frames.numel() - 1)
    return valid_frames[positions]


def select_z_indices(d_count: int, mode: str, slice_count: int) -> list[int]:
    if d_count <= 0:
        raise ValueError("Volume has no z-slices.")
    if mode == "all":
        return list(range(d_count))
    if mode == "center":
        return [d_count // 2]
    count = max(1, min(int(slice_count), d_count))
    if count == 1:
        return [d_count // 2]
    return sorted({int(round(v)) for v in np.linspace(0, d_count - 1, count)})


def sample_flat_values(tensor: torch.Tensor, rng: np.random.Generator, limit: int) -> np.ndarray:
    flat = tensor.detach().float().cpu().numpy().reshape(-1)
    if flat.size <= limit:
        return flat
    indices = rng.choice(flat.size, size=limit, replace=False)
    return flat[indices]


def sample_token_vectors(
    z: torch.Tensor,
    indices: torch.Tensor,
    rng: np.random.Generator,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = z.detach().float().permute(0, *range(2, z.ndim), 1).reshape(-1, z.shape[1]).cpu().numpy()
    codes = indices.detach().reshape(-1).cpu().numpy()
    if tokens.shape[0] <= limit:
        return tokens, codes
    chosen = rng.choice(tokens.shape[0], size=limit, replace=False)
    return tokens[chosen], codes[chosen]


@torch.no_grad()
def reconstruct_volume(
    model: VQVAE,
    volume_thwd: torch.Tensor,
    batch_size: int,
    rng: np.random.Generator,
    tokens_per_subject: int,
    num_embeddings: int,
    valid_frames: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    d_count = int(volume_thwd.shape[3])
    recon_chunks = []
    code_counts = torch.zeros(num_embeddings, dtype=torch.long)
    latent_sum = 0.0
    latent_sumsq = 0.0
    latent_count = 0
    value_samples = []
    vector_samples = []
    code_samples = []
    sampled_tokens = 0

    for start in range(0, d_count, batch_size):
        end = min(start + batch_size, d_count)
        chunk = volume_thwd[:, :, :, start:end]
        batch = chunk.permute(3, 1, 2, 0).unsqueeze(1).contiguous()

        z = model.encode(batch)
        quantized, _, indices = model.quantize(z)
        recon = model.decode(quantized).cpu()
        recon_chunks.append(recon)

        latent_valid = project_temporal_validity(valid_frames, int(indices.shape[-1]))
        if not bool(latent_valid.any()):
            raise ValueError("No valid latent temporal tokens remain after projection.")
        valid_indices = indices[..., latent_valid]
        code_counts += torch.bincount(valid_indices.reshape(-1).cpu(), minlength=num_embeddings)
        z_float = z.detach().float()[..., latent_valid]
        latent_sum += float(z_float.sum().item())
        latent_sumsq += float((z_float * z_float).sum().item())
        latent_count += int(z_float.numel())

        if sum(v.size for v in value_samples) < 200_000:
            value_samples.append(sample_flat_values(z_float, rng, 16_384))

        remaining = max(0, tokens_per_subject - sampled_tokens)
        if remaining:
            vecs, codes = sample_token_vectors(z_float, valid_indices, rng, min(remaining, 64))
            vector_samples.append(vecs)
            code_samples.append(codes)
            sampled_tokens += int(vecs.shape[0])

    recon_bdchwt = torch.cat(recon_chunks, dim=0)
    recon_thwd = recon_bdchwt[:, 0].permute(3, 1, 2, 0).contiguous()

    stats = {
        "code_counts": code_counts,
        "latent_sum": latent_sum,
        "latent_sumsq": latent_sumsq,
        "latent_count": latent_count,
        "value_samples": np.concatenate(value_samples) if value_samples else np.empty((0,), dtype=np.float32),
        "vector_samples": np.concatenate(vector_samples, axis=0) if vector_samples else np.empty((0, 0), dtype=np.float32),
        "code_samples": np.concatenate(code_samples, axis=0) if code_samples else np.empty((0,), dtype=np.int64),
    }
    return recon_thwd, stats


def save_recon_examples(examples: list[tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    if not examples:
        return
    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 2, figsize=(4.0, 2.0 * n_rows), squeeze=False)
    for row, (inp, rec) in enumerate(examples):
        for col, image in enumerate((inp, rec)):
            axes[row, col].imshow(image, cmap="gray", vmin=-1.0, vmax=1.0)
            axes[row, col].set_axis_off()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def display_name(run_name: str) -> str:
    return DISPLAY_NAMES.get(run_name, run_name)


def codebook_stats(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    total = max(float(counts.sum()), 1.0)
    probs = counts.astype(np.float64) / total
    nonzero = probs > 0
    used_codes = int(nonzero.sum())
    num_embeddings = int(counts.size)
    sorted_probs = np.sort(probs)[::-1]
    perplexity = float(np.exp(-np.sum(probs[nonzero] * np.log(probs[nonzero])))) if nonzero.any() else 0.0
    return {
        "num_embeddings": num_embeddings,
        "used_codes": used_codes,
        "dead_codes": num_embeddings - used_codes,
        "usage_ratio": used_codes / max(num_embeddings, 1),
        "code_perplexity": perplexity,
        "normalized_perplexity": perplexity / max(num_embeddings, 1),
        "top1_code_fraction": float(sorted_probs[0]) if sorted_probs.size else 0.0,
        "top10_code_fraction": float(sorted_probs[:10].sum()) if sorted_probs.size else 0.0,
        "top100_code_fraction": float(sorted_probs[:100].sum()) if sorted_probs.size else 0.0,
    }


def save_hist(values: np.ndarray, path: Path, run_name: str = "") -> None:
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(values, bins=80, color="#3b6ea8", alpha=0.9)
    ax.set_title(f"{display_name(run_name)} latent values", fontsize=10)
    ax.set_xlabel("encoder latent value")
    ax.set_ylabel("sampled tokens")
    ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_code_usage(counts: np.ndarray, path: Path, run_name: str = "") -> None:
    total = max(float(np.sum(counts)), 1.0)
    ranked = np.sort(np.asarray(counts, dtype=np.float64))[::-1] / total
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(np.arange(1, ranked.size + 1), ranked, linewidth=1.0, color="#262626")
    ax.set_yscale("log")
    ax.set_title(f"{display_name(run_name)} code usage rank", fontsize=10)
    ax.set_xlabel("code rank by frequency")
    ax.set_ylabel("token fraction, log scale")
    ax.text(
        0.98,
        0.95,
        "flatter = less collapse",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_codebook_analysis(counts: np.ndarray, path: Path, run_name: str = "") -> None:
    counts = np.asarray(counts, dtype=np.int64)
    stats = codebook_stats(counts)
    total = max(float(counts.sum()), 1.0)
    ranked = np.sort(counts.astype(np.float64))[::-1] / total
    cumulative = np.cumsum(ranked)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    fig.suptitle(f"{display_name(run_name)} codebook diagnostics", fontsize=12)

    ax = axes[0, 0]
    bars = ax.bar(["used", "dead"], [stats["used_codes"], stats["dead_codes"]], color=["#2f6f77", "#b8b8b8"])
    ax.set_title("Used codes (higher is better)", fontsize=10)
    ax.set_ylabel("codes")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[0, 1]
    ax.plot(np.arange(1, ranked.size + 1), ranked, color="#262626", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("Usage concentration (flatter is better)", fontsize=10)
    ax.set_xlabel("code rank")
    ax.set_ylabel("token fraction")

    ax = axes[1, 0]
    ax.plot(np.arange(1, cumulative.size + 1), cumulative, color="#6f4e7c", linewidth=1.0)
    ax.axhline(0.8, color="#9a9a9a", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_title("Dominance by top codes (slower rise is better)", fontsize=10)
    ax.set_xlabel("top-k codes")
    ax.set_ylabel("cumulative token fraction")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"effective codes: {stats['code_perplexity']:.1f}",
        f"effective / total: {100.0 * stats['normalized_perplexity']:.1f}%",
        f"used / total: {100.0 * stats['usage_ratio']:.1f}%",
        f"top 1 code share: {100.0 * stats['top1_code_fraction']:.1f}%",
        f"top 10 code share: {100.0 * stats['top10_code_fraction']:.1f}%",
        "",
        "good: many effective codes",
        "bad: few codes dominate",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10, family="monospace")

    for ax in axes.flat[:3]:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_tsne(vectors: np.ndarray, codes: np.ndarray, path: Path, seed: int, run_name: str = "") -> None:
    if vectors.shape[0] < 10:
        return
    n = vectors.shape[0]
    perplexity = max(5, min(30, (n - 1) // 3))
    xy = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    color = np.log1p(codes.astype(np.float32)) if codes.size == n else "#2f6f77"
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c=color, cmap="viridis", alpha=0.65, linewidths=0)
    ax.set_title(f"{display_name(run_name)} latent t-SNE", fontsize=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(run_name: str, rows: list[dict[str, str]], args: argparse.Namespace, output_dir: Path) -> dict:
    config_path = Path(args.config_dir) / f"{run_name}.yaml"
    checkpoint_path = Path(args.checkpoint_root) / run_name / args.checkpoint_name
    run_dir = output_dir / run_name
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    transform = build_eval_transform(config)
    model, ckpt, stage1_identity = load_model(config_path, checkpoint_path)
    num_embeddings = int(config.model.params.num_embeddings)
    rng = np.random.default_rng(args.seed)

    total_sse = 0.0
    total_abs = 0.0
    total_voxels = 0
    total_eval_slices = 0
    total_ssim_sum = 0.0
    total_ssim_count = 0
    subject_rows = []
    examples = []
    all_code_counts = torch.zeros(num_embeddings, dtype=torch.long)
    latent_sum = 0.0
    latent_sumsq = 0.0
    latent_count = 0
    latent_value_samples = []
    latent_vectors = []
    latent_vector_codes = []

    print(f"\n[{run_name}] checkpoint={checkpoint_path}", flush=True)
    for subject_idx, row in enumerate(rows, start=1):
        image_path = row["image"]
        data = transform({"image": image_path})
        volume = torch.as_tensor(data["image"]).float()
        if volume.ndim != 4:
            raise ValueError(f"Expected (T,H,W,D), got {tuple(volume.shape)} for {image_path}")
        z_indices = select_z_indices(int(volume.shape[3]), args.slice_mode, args.slice_count)
        eval_volume = volume[:, :, :, z_indices].contiguous()
        if "temporal_valid_mask" in data:
            valid_frames = torch.as_tensor(data["temporal_valid_mask"], dtype=torch.bool)
        else:
            original_length = int(
                torch.as_tensor(data.get("original_temporal_length", volume.shape[0])).item()
            )
            valid_frames = torch.arange(eval_volume.shape[0]) < min(
                original_length, eval_volume.shape[0]
            )

        recon, stats = reconstruct_volume(
            model=model,
            volume_thwd=eval_volume,
            batch_size=args.batch_size,
            rng=rng,
            tokens_per_subject=args.tokens_per_subject,
            num_embeddings=num_embeddings,
            valid_frames=valid_frames,
        )
        if tuple(recon.shape) != tuple(eval_volume.shape):
            raise ValueError(
                f"Reconstruction shape mismatch for {image_path}: "
                f"input={tuple(eval_volume.shape)} recon={tuple(recon.shape)}"
            )

        metrics = masked_reconstruction_metrics(eval_volume, recon, valid_frames)
        sse = metrics["sse"]
        abs_sum = metrics["absolute_error"]
        n_vox = metrics["voxels"]
        mse = metrics["mse"]
        mae = metrics["mae"]
        ssim_mean, ssim_count = metrics["ssim"], metrics["ssim_count"]

        total_sse += sse
        total_abs += abs_sum
        total_voxels += n_vox
        total_eval_slices += len(z_indices)
        total_ssim_sum += ssim_mean * ssim_count
        total_ssim_count += ssim_count

        all_code_counts += stats["code_counts"]
        latent_sum += stats["latent_sum"]
        latent_sumsq += stats["latent_sumsq"]
        latent_count += stats["latent_count"]
        latent_value_samples.append(stats["value_samples"])
        if stats["vector_samples"].size:
            latent_vectors.append(stats["vector_samples"])
            latent_vector_codes.append(stats["code_samples"])

        subject_rows.append(
            {
                "run": run_name,
                "subject": Path(image_path).name,
                "input_shape_T_H_W_D": json.dumps(list(volume.shape)),
                "eval_z_indices": json.dumps(z_indices),
                "n_eval_slices": len(z_indices),
                "n_valid_frames": metrics["valid_frames"],
                "mse": mse,
                "mae": mae,
                "rmse": math.sqrt(mse),
                "psnr": mse_to_psnr(mse),
                "ssim": ssim_mean,
            }
        )

        if len(examples) < args.num_examples:
            valid_indices = valid_frames.nonzero(as_tuple=True)[0]
            t_mid = int(valid_indices[len(valid_indices) // 2])
            d_mid = eval_volume.shape[3] // 2
            examples.append((eval_volume[t_mid, :, :, d_mid].numpy(), recon[t_mid, :, :, d_mid].numpy()))

        print(
            f"[{run_name}] {subject_idx:03d}/{len(rows)} "
            f"z={z_indices} mse={mse:.6f} psnr={mse_to_psnr(mse):.2f} ssim={ssim_mean:.4f}",
            flush=True,
        )

    value_samples = np.concatenate(latent_value_samples) if latent_value_samples else np.empty((0,), dtype=np.float32)
    if value_samples.size > args.latent_value_samples:
        value_samples = value_samples[rng.choice(value_samples.size, size=args.latent_value_samples, replace=False)]

    vectors = np.concatenate(latent_vectors, axis=0) if latent_vectors else np.empty((0, 0), dtype=np.float32)
    vector_codes = np.concatenate(latent_vector_codes, axis=0) if latent_vector_codes else np.empty((0,), dtype=np.int64)
    if vectors.shape[0] > args.latent_vector_samples:
        chosen = rng.choice(vectors.shape[0], size=args.latent_vector_samples, replace=False)
        vectors = vectors[chosen]
        vector_codes = vector_codes[chosen]

    counts_np = all_code_counts.numpy()
    cb_stats = codebook_stats(counts_np)

    latent_mean = latent_sum / max(latent_count, 1)
    latent_var = max(latent_sumsq / max(latent_count, 1) - latent_mean * latent_mean, 0.0)
    summary = {
        "run": run_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_epoch": ckpt.get("epoch", ""),
        "checkpoint_best_loss": ckpt.get("best_loss", ""),
        "stage1_identity": stage1_identity,
        "n_subjects": len(rows),
        "split_name": args.split_name,
        "all_valid_frames": True,
        "all_slices": args.slice_mode == "all",
        "slice_mode": args.slice_mode,
        "slice_count": args.slice_count,
        "n_eval_slices": total_eval_slices,
        "n_voxels": total_voxels,
        "mse": total_sse / total_voxels,
        "mae": total_abs / total_voxels,
        "rmse": math.sqrt(total_sse / total_voxels),
        "psnr": mse_to_psnr(total_sse / total_voxels),
        "ssim": total_ssim_sum / max(total_ssim_count, 1),
        "latent_mean": latent_mean,
        "latent_std": math.sqrt(latent_var),
        **cb_stats,
        **rate_metrics(
            int(counts_np.sum()),
            num_embeddings,
            total_voxels,
            args.input_bits_per_voxel,
        ),
    }
    if args.downstream_metrics_json:
        downstream = json.loads(Path(args.downstream_metrics_json).read_text())
        summary["downstream_generation_metrics"] = downstream
        summary["downstream_metrics_source"] = file_identity(args.downstream_metrics_json)
        summary["downstream_metrics_status"] = "reported"
    else:
        summary["downstream_generation_metrics"] = None
        summary["downstream_metrics_status"] = "not_provided"

    write_csv(run_dir / "metrics_by_subject.csv", subject_rows)
    save_recon_examples(examples, plot_dir / "recon_examples.png")
    save_hist(value_samples, plot_dir / "latent_value_hist.png", run_name)
    save_code_usage(counts_np, plot_dir / "code_usage_sorted.png", run_name)
    save_codebook_analysis(counts_np, plot_dir / "codebook_analysis.png", run_name)
    save_tsne(vectors, vector_codes, plot_dir / "latent_tsne.png", args.seed, run_name)

    np.savez_compressed(
        run_dir / "latent_samples.npz",
        values=value_samples.astype(np.float32),
        vectors=vectors.astype(np.float32),
        codes=vector_codes.astype(np.int64),
        code_counts=counts_np.astype(np.int64),
    )
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_comparison_plot(summary_rows: list[dict], output_dir: Path) -> None:
    names = [display_name(row["run"]) for row in summary_rows]
    metrics = [
        ("psnr", "PSNR, dB\nhigher is better"),
        ("ssim", "SSIM\nhigher is better"),
        ("mae", "MAE\nlower is better"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 3.6))
    for ax, (metric, label) in zip(axes, metrics):
        values = [float(row[metric]) for row in summary_rows]
        ax.bar(np.arange(len(names)), values, color="#404040", width=0.65)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_codebook_comparison(summary_rows: list[dict], output_dir: Path) -> None:
    names = [display_name(row["run"]) for row in summary_rows]
    metrics = [
        ("usage_ratio", "used code ratio\nhigher is better"),
        ("normalized_perplexity", "effective code ratio\nhigher is better"),
        ("top10_code_fraction", "top-10 token share\nlower is better"),
        ("dead_codes", "dead codes\nlower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for ax, (metric, label) in zip(axes.flat, metrics):
        values = [float(row.get(metric, 0.0)) for row in summary_rows]
        ax.bar(np.arange(len(names)), values, color="#2f6f77", width=0.65)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "codebook_comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def rank_rows(summary_rows: list[dict]) -> list[dict]:
    indexed = list(enumerate(summary_rows))

    def ranks(metric: str, reverse: bool) -> dict[int, int]:
        ordered = sorted(indexed, key=lambda item: float(item[1][metric]), reverse=reverse)
        return {idx: rank + 1 for rank, (idx, _) in enumerate(ordered)}

    psnr_rank = ranks("psnr", True)
    ssim_rank = ranks("ssim", True)
    mae_rank = ranks("mae", False)
    ranked = []
    for idx, row in indexed:
        mean_rank = (psnr_rank[idx] + ssim_rank[idx] + mae_rank[idx]) / 3.0
        ranked.append(
            {
                "run": row["run"],
                "checkpoint_name": row.get("checkpoint_name", ""),
                "mean_reconstruction_rank": mean_rank,
                "psnr_rank": psnr_rank[idx],
                "ssim_rank": ssim_rank[idx],
                "mae_rank": mae_rank[idx],
                "psnr": row["psnr"],
                "ssim": row["ssim"],
                "mae": row["mae"],
                "code_perplexity": row.get("code_perplexity", ""),
                "usage_ratio": row.get("usage_ratio", ""),
                "top10_code_fraction": row.get("top10_code_fraction", ""),
            }
        )
    return sorted(ranked, key=lambda row: float(row["mean_reconstruction_rank"]))


def regenerate_plots(output_dir: Path, runs: list[str], seed: int) -> list[dict]:
    summaries = []
    for run_name in runs:
        run_dir = output_dir / run_name
        summary_path = run_dir / "summary.json"
        samples_path = run_dir / "latent_samples.npz"
        if not summary_path.exists() or not samples_path.exists():
            continue

        with summary_path.open() as f:
            summary = json.load(f)
        if "checkpoint_name" not in summary and summary.get("checkpoint"):
            summary["checkpoint_name"] = Path(summary["checkpoint"]).name
        samples = np.load(samples_path)
        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        counts = samples["code_counts"]
        summary.update(codebook_stats(counts))
        save_hist(samples["values"], plot_dir / "latent_value_hist.png", run_name)
        save_code_usage(counts, plot_dir / "code_usage_sorted.png", run_name)
        save_codebook_analysis(counts, plot_dir / "codebook_analysis.png", run_name)
        save_tsne(samples["vectors"], samples["codes"], plot_dir / "latent_tsne.png", seed, run_name)

        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)

    if summaries:
        write_csv(output_dir / "metrics_summary.csv", summaries)
        write_csv(output_dir / "ranking.csv", rank_rows(summaries))
        save_comparison_plot(summaries, output_dir)
        save_codebook_comparison(summaries, output_dir)
    return summaries
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.split_name != "test" and not args.allow_non_test_split:
        raise ValueError("Official Stage-1 evaluation requires --split_name test.")
    if args.slice_mode != "all" and not args.diagnostic_subset:
        raise ValueError("Official Stage-1 evaluation requires --slice_mode all.")
    if args.downstream_metrics_json is None and not args.allow_missing_downstream and not args.plots_only:
        raise ValueError(
            "Provide --downstream_metrics_json for the official report or explicitly "
            "use --allow_missing_downstream."
        )
    torch.set_num_threads(max(1, int(args.num_threads)))
    torch.set_num_interop_threads(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = args.run if args.run else list(RUN_NAMES)
    if args.plots_only:
        summaries = regenerate_plots(output_dir, runs, args.seed)
        print(f"Regenerated plots for {len(summaries)} runs under: {output_dir}", flush=True)
        return

    rows = load_rows(Path(args.csv), args.max_subjects)
    print(f"Loaded {len(rows)} held-out subjects from {args.csv}", flush=True)
    print(f"CPU threads: {torch.get_num_threads()} | CUDA available to script: {torch.cuda.is_available()}", flush=True)

    summaries = []
    for run_name in runs:
        summaries.append(evaluate_run(run_name, rows, args, output_dir))

    write_csv(output_dir / "metrics_summary.csv", summaries)
    write_csv(output_dir / "ranking.csv", rank_rows(summaries))
    save_comparison_plot(summaries, output_dir)
    save_codebook_comparison(summaries, output_dir)
    print(f"\nSaved summary: {output_dir / 'metrics_summary.csv'}", flush=True)
    print(f"Saved plots under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

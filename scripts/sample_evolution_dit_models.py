#!/usr/bin/env python
"""
Sample synthetic 4D CMR volumes from the MNM2 evolution DiT checkpoints.

The wrapper maps each DiT run to the VQ-GAN decoder that produced its training
latents, resolves the matching DiT config/checkpoint, and delegates generation
to src/scripts/sample_dit.py.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()
sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf
from src.utils.checkpointing import adapt_legacy_checkpoint
from src.utils.sample_integrity import resolve_checkpoint_reference


SAMPLE_SCRIPT = REPO_ROOT / "src" / "scripts" / "sample_dit.py"

LEGACY_FIXED_STAGE1_CFG = (
    Path(os.environ.get("MNM2_FIXED_STAGE1_CFG", ""))
    if os.environ.get("MNM2_FIXED_STAGE1_CFG")
    else RUNS_ROOT
    / "legacy_fixed"
    / "outputs"
    / "vqgan"
    / "vqgan_ds4_all_dims_wide_e16_mnm2_224_d12_ddp.yaml"
)
LEGACY_FIXED_STAGE1_CKPT = (
    Path(os.environ.get("MNM2_FIXED_STAGE1_CKPT", ""))
    if os.environ.get("MNM2_FIXED_STAGE1_CKPT")
    else RUNS_ROOT
    / "legacy_fixed"
    / "outputs"
    / "vqgan"
    / "last_checkpoint.pth"
)

NEW_STAGE1_FAMILIES = (
    "S1_ds8xy_noT_native",
    "S1_ds4xy_noT_native",
    "S1_ds4_all_dims_paddiv",
)


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    diff_cfg: Path | None
    diff_ckpt: Path
    stage1_cfg: Path
    stage1_ckpt: Path
    scale_factor: float
    latent_shape: list[int]
    samplers: list["SamplerSpec"]


@dataclass(frozen=True)
class SamplerSpec:
    label: str
    scheduler: str
    timesteps: int


@dataclass(frozen=True)
class DecoderSpec:
    label: str
    skip_quantization: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample all available MNM2 evolution DiT checkpoints."
    )
    parser.add_argument("--runs_root", default=str(RUNS_ROOT / "outputs" / "dit"))
    parser.add_argument("--output_root", default=str(RUNS_ROOT / "outputs" / "synthetic_samples"))
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of latent trajectories evaluated per DiT forward pass.",
    )
    parser.add_argument(
        "--per_sample_output_dirs",
        action="store_true",
        help=(
            "Load each checkpoint once while retaining resumable sample_NNN/ "
            "output directories and completion manifests."
        ),
    )
    parser.add_argument(
        "--flat_output_layout",
        action="store_true",
        help="Write quantized sample_NNN.nii.gz files directly under RUN/CHECKPOINT/.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help="Inference steps for a single --scheduler run.",
    )
    parser.add_argument(
        "--scheduler",
        default="fast",
        choices=["fast", "trained", "auto", "ddpm", "ddim", "dpm_pp", "flow_matching"],
        help=(
            "Single-sampler mode. 'fast' uses dpm_pp for DDPM configs and "
            "flow_matching for flow configs. 'trained' uses the training "
            "scheduler type, full scheduler.num_train_timesteps for DDPM, "
            "and --flow_matching_timesteps for flow configs. Ignored when "
            "--samplers is set."
        ),
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=None,
        choices=[
            "trained",
            "full_ddpm",
            "dpm_pp",
            "fast",
            "auto",
            "ddpm",
            "ddim",
            "flow_matching",
        ],
        help=(
            "Sampler grid. Use 'trained' to follow each config's training "
            "scheduler: full DDPM chain for DDPM configs and configured flow "
            "NFEs for flow configs. Use 'full_ddpm dpm_pp' for the full DDPM "
            "chain plus one DPM++ run per DDPM checkpoint."
        ),
    )
    parser.add_argument(
        "--ddpm_timesteps",
        type=int,
        default=None,
        help="Timesteps for full_ddpm. Defaults to scheduler.num_train_timesteps from each config.",
    )
    parser.add_argument(
        "--dpmpp_timesteps",
        type=int,
        default=50,
        help="Timesteps for the dpm_pp sampler grid entry.",
    )
    parser.add_argument(
        "--flow_matching_timesteps",
        type=int,
        default=50,
        help=(
            "Euler ODE steps / NFEs for flow-matching configs when using "
            "--scheduler trained or --samplers trained."
        ),
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="Sample every *.pth checkpoint in each run directory.",
    )
    parser.add_argument(
        "--include_final_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include final_model.pth in --all_checkpoints. Weight availability "
            "is inspected by sample_dit; it is never inferred from the filename."
        ),
    )
    parser.add_argument(
        "--checkpoint_glob",
        default="*.pth",
        help="Checkpoint glob used with --all_checkpoints.",
    )
    parser.add_argument("--checkpoint_name", default="best_model.pth")
    parser.add_argument("--fallback_checkpoint_name", default="last_checkpoint.pth")
    parser.add_argument("--weights", default="ema", choices=["ema", "raw", "both"])
    parser.add_argument(
        "--decoder_modes",
        nargs="+",
        default=["both"],
        choices=["quantized", "direct", "both"],
        help=(
            "Decoder modes to sample. 'quantized' snaps generated latents to "
            "the VQ codebook before decoding; 'direct' passes latents directly "
            "to the decoder."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional run-name filter. Accepts exact names or substrings.",
    )
    parser.add_argument(
        "--output_layout",
        default="4d",
        choices=["4d", "frames", "both"],
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility flag; sample_dit skips only a hash-valid completion manifest.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even when outputs exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    return parser.parse_args()


def choose_checkpoint(run_dir: Path, primary: str, fallback: str) -> Path | None:
    primary_path = run_dir / primary
    if primary_path.is_file():
        return resolve_checkpoint_reference(primary_path)
    fallback_path = run_dir / fallback
    if fallback_path.is_file():
        return resolve_checkpoint_reference(fallback_path)
    pointer = run_dir / "last_checkpoint.json"
    if pointer.is_file():
        return resolve_checkpoint_reference(pointer)
    return None


def checkpoint_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.name
    if name == "best_model.pth":
        return (0, 0)
    match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pth", name)
    if match:
        return (1, int(match.group(1)))
    match = re.fullmatch(r"checkpoint_update_(\d+)\.pth", name)
    if match:
        return (1, int(match.group(1)))
    if name == "final_model.pth":
        return (2, 0)
    if name == "last_checkpoint.pth":
        return (3, 0)
    return (4, name)


def checkpoints_for_run(run_dir: Path, args: argparse.Namespace) -> list[Path]:
    if args.all_checkpoints:
        checkpoints = [
            resolve_checkpoint_reference(path)
            for path in run_dir.glob(args.checkpoint_glob)
        ]
        checkpoints = sorted(set(checkpoints), key=checkpoint_sort_key)
        if not args.include_final_model:
            checkpoints = [path for path in checkpoints if path.name != "final_model.pth"]
        return checkpoints

    checkpoint = choose_checkpoint(
        run_dir, args.checkpoint_name, args.fallback_checkpoint_name
    )
    return [] if checkpoint is None else [checkpoint]


def find_diff_config(run_name: str) -> Path | None:
    config_name = run_name.removesuffix("_retrain_300k")
    candidates = [
        REPO_ROOT / "configs" / "dit" / "fixed" / f"{config_name}.yaml",
        REPO_ROOT / "configs" / "dit" / "native_padded" / f"{config_name}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def sampling_config_for_checkpoint(
    checkpoint_path: Path,
    fallback_config_path: Path,
):
    """Prefer the immutable config embedded in modern checkpoints.

    A run may span checkpoints written before and after operational settings
    such as evaluation or checkpoint cadence changed. Passing the current
    external YAML for those checkpoints causes the strict provenance check in
    sample_dit.py to reject an otherwise valid inference artifact.
    """
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    checkpoint = adapt_legacy_checkpoint(payload)
    resolved_config = checkpoint.get("resolved_config")
    if not checkpoint.get("legacy") and isinstance(resolved_config, Mapping):
        return OmegaConf.create(resolved_config), None
    return OmegaConf.load(fallback_config_path), fallback_config_path


def stage1_for_run(run_name: str) -> tuple[Path, Path, str]:
    if run_name.startswith("F"):
        return LEGACY_FIXED_STAGE1_CFG, LEGACY_FIXED_STAGE1_CKPT, "legacy_fixed_wide_e16"

    for family in NEW_STAGE1_FAMILIES:
        if run_name.startswith(family):
            cfg = REPO_ROOT / "configs" / "stage1" / f"{family}.yaml"
            ckpt_dir = RUNS_ROOT / "outputs" / "stage1" / family
            ckpt = ckpt_dir / "best_model.pth"
            if not ckpt.is_file():
                ckpt = ckpt_dir / "last_checkpoint.pth"
            return cfg, ckpt, family

    raise ValueError(f"Cannot infer Stage 1 family for run: {run_name}")


def scheduler_for_request(requested: str, diff_cfg) -> str:
    scheduler_type = str(diff_cfg.get("scheduler_type", "ddpm")).lower()
    if requested == "fast":
        return "flow_matching" if scheduler_type == "flow_matching" else "dpm_pp"
    if requested == "auto":
        return "auto"
    return requested


def num_train_timesteps(diff_cfg) -> int:
    return int(diff_cfg.scheduler.get("num_train_timesteps", 1000))


def sampler_specs_for_config(
    args: argparse.Namespace, diff_cfg
) -> tuple[list[SamplerSpec], list[str]]:
    requested = args.samplers or [args.scheduler]
    scheduler_type = str(diff_cfg.get("scheduler_type", "ddpm")).lower()
    is_flow_matching = scheduler_type == "flow_matching"
    samplers: list[SamplerSpec] = []
    skipped: list[str] = []

    for item in requested:
        if item == "trained":
            if is_flow_matching:
                steps = int(args.flow_matching_timesteps)
                samplers.append(
                    SamplerSpec(
                        f"trained_flow_matching_{steps}steps",
                        "flow_matching",
                        steps,
                    )
                )
            else:
                steps = num_train_timesteps(diff_cfg)
                samplers.append(SamplerSpec(f"trained_ddpm_{steps}steps", "ddpm", steps))
        elif item == "full_ddpm":
            if is_flow_matching:
                skipped.append("full_ddpm is incompatible with scheduler_type=flow_matching")
                continue
            steps = args.ddpm_timesteps or num_train_timesteps(diff_cfg)
            samplers.append(SamplerSpec(f"full_ddpm_{steps}steps", "ddpm", steps))
        elif item == "dpm_pp":
            if is_flow_matching:
                skipped.append("dpm_pp is incompatible with scheduler_type=flow_matching")
                continue
            steps = int(args.dpmpp_timesteps)
            samplers.append(SamplerSpec(f"dpm_pp_{steps}steps", "dpm_pp", steps))
        elif item == "fast":
            scheduler = scheduler_for_request("fast", diff_cfg)
            steps = args.timesteps
            samplers.append(SamplerSpec(f"{scheduler}_{steps}steps", scheduler, steps))
        else:
            scheduler = scheduler_for_request(item, diff_cfg)
            steps = args.timesteps
            samplers.append(SamplerSpec(f"{scheduler}_{steps}steps", scheduler, steps))

    return samplers, skipped


def decoder_specs_for_args(args: argparse.Namespace) -> list[DecoderSpec]:
    requested = list(args.decoder_modes)
    if "both" in requested:
        requested = ["quantized", "direct"]

    specs: list[DecoderSpec] = []
    seen: set[str] = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        if item == "quantized":
            specs.append(DecoderSpec("decode_quantized", False))
        elif item == "direct":
            specs.append(DecoderSpec("decode_direct", True))
        else:
            raise ValueError(f"Unknown decoder mode {item!r}")
    return specs


def latent_shape_from_config(diff_cfg) -> list[int]:
    params = diff_cfg.model.params
    return [int(params.in_channels), *[int(v) for v in params.input_size]]


def completed_samples(out_dir: Path) -> int:
    return len(list(out_dir.glob("sample_*.nii.gz")))


def selected(run_name: str, filters: list[str] | None) -> bool:
    if not filters:
        return True
    return any(item == run_name or item in run_name for item in filters)


def discover_runs(args: argparse.Namespace) -> tuple[list[RunSpec], list[str]]:
    runs_root = Path(args.runs_root)
    specs: list[RunSpec] = []
    skipped: list[str] = []

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        run_name = run_dir.name
        if not selected(run_name, args.only):
            continue

        diff_cfg_path = find_diff_config(run_name)
        if diff_cfg_path is None:
            skipped.append(f"{run_name}: no matching config under configs/dit")
            continue

        checkpoints = checkpoints_for_run(run_dir, args)
        if not checkpoints:
            skipped.append(
                f"{run_name}: no checkpoints matching "
                f"{args.checkpoint_glob if args.all_checkpoints else args.checkpoint_name}"
            )
            continue

        try:
            stage1_cfg, stage1_ckpt, stage1_label = stage1_for_run(run_name)
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        missing = [str(p) for p in (stage1_cfg, stage1_ckpt) if not p.is_file()]
        if missing:
            skipped.append(
                f"{run_name}: missing Stage 1 inputs for {stage1_label}: "
                + ", ".join(missing)
            )
            continue

        for diff_ckpt in checkpoints:
            diff_cfg, command_diff_cfg_path = sampling_config_for_checkpoint(
                diff_ckpt,
                diff_cfg_path,
            )
            samplers, sampler_skips = sampler_specs_for_config(args, diff_cfg)
            for item in sampler_skips:
                skipped.append(f"{run_name}/{diff_ckpt.name}: {item}")
            if not samplers:
                continue
            specs.append(
                RunSpec(
                    run_name=run_name,
                    diff_cfg=command_diff_cfg_path,
                    diff_ckpt=diff_ckpt,
                    stage1_cfg=stage1_cfg,
                    stage1_ckpt=stage1_ckpt,
                    scale_factor=float(diff_cfg.training.get("scale_factor", 1.0)),
                    latent_shape=latent_shape_from_config(diff_cfg),
                    samplers=samplers,
                )
            )

    return specs, skipped


def output_dir_for(
    spec: RunSpec,
    sampler: SamplerSpec,
    decoder: DecoderSpec,
    args: argparse.Namespace,
) -> Path:
    return (
        Path(args.output_root)
        / spec.run_name
        / spec.diff_ckpt.stem
        / args.weights
        / decoder.label
        / sampler.label
    )


def combined_decoder_mode(args: argparse.Namespace) -> str:
    requested = set(args.decoder_modes)
    if "both" in requested or requested == {"direct", "quantized"}:
        return "both"
    if requested == {"direct"}:
        return "direct"
    if requested == {"quantized"}:
        return "quantized"
    raise ValueError(f"Unsupported decoder mode combination: {sorted(requested)}")


def combined_output_dir_for(
    spec: RunSpec,
    sampler: SamplerSpec,
    args: argparse.Namespace,
    sample_index: int | None = None,
) -> Path:
    if getattr(args, "flat_output_layout", False):
        if sample_index is not None:
            raise ValueError("Flat output layout operates on all sample indices together.")
        return Path(args.output_root) / spec.run_name / spec.diff_ckpt.stem
    root = (
        Path(args.output_root)
        / spec.run_name
        / spec.diff_ckpt.stem
        / args.weights
        / sampler.label
    )
    return root if sample_index is None else root / f"sample_{sample_index:03d}"


def combined_command_for(
    spec: RunSpec,
    sampler: SamplerSpec,
    args: argparse.Namespace,
    sample_index: int | None = None,
) -> list[str]:
    cmd = command_for(
        spec,
        sampler,
        DecoderSpec("combined", False),
        args,
    )
    output_index = cmd.index("--output_dir") + 1
    cmd[output_index] = str(combined_output_dir_for(spec, sampler, args, sample_index))
    if sample_index is not None:
        count_index = cmd.index("--n_samples") + 1
        cmd[count_index] = "1"
        cmd.extend(["--sample_index_start", str(sample_index)])
    if getattr(args, "per_sample_output_dirs", False):
        cmd.append("--per_sample_output_dirs")
    if getattr(args, "flat_output_layout", False):
        cmd.append("--flat_sample_files")
    cmd.extend(["--decoder_mode", combined_decoder_mode(args)])
    if args.force:
        cmd.append("--force")
    return cmd


def command_for(
    spec: RunSpec,
    sampler: SamplerSpec,
    decoder: DecoderSpec,
    args: argparse.Namespace,
) -> list[str]:
    out_dir = output_dir_for(spec, sampler, decoder, args)
    cmd = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        "--stage1_cfg",
        str(spec.stage1_cfg),
        "--stage1_ckpt",
        str(spec.stage1_ckpt),
        "--diff_ckpt",
        str(spec.diff_ckpt),
        "--output_dir",
        str(out_dir),
        "--n_samples",
        str(args.n_samples),
        "--batch_size",
        str(getattr(args, "batch_size", 1)),
        "--timesteps",
        str(sampler.timesteps),
        "--scheduler",
        sampler.scheduler,
        "--scale_factor",
        str(spec.scale_factor),
        "--weights",
        args.weights,
        "--seed",
        str(args.seed),
        "--latent_shape",
        *[str(v) for v in spec.latent_shape],
        "--output_layout",
        args.output_layout,
        "--device",
        args.device,
    ]
    if spec.diff_cfg is not None:
        cmd.extend(["--diff_cfg", str(spec.diff_cfg)])
    if decoder.skip_quantization:
        cmd.append("--skip_decoder_quantization")
    return cmd


def main() -> int:
    args = parse_args()
    specs, skipped = discover_runs(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if skipped:
        print("Skipped runs:")
        for item in skipped:
            print(f"  - {item}")

    if not specs:
        print("No runnable DiT checkpoints found.")
        return 1

    shard_count = int(getattr(args, "shard_count", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count.")
    if (
        getattr(args, "per_sample_output_dirs", False)
        or getattr(args, "flat_output_layout", False)
    ):
        tasks = [
            (spec, sampler, None)
            for spec in specs
            for sampler in spec.samplers
        ]
    else:
        tasks = [
            (spec, sampler, sample_index)
            for spec in specs
            for sampler in spec.samplers
            for sample_index in range(args.n_samples)
        ]
    tasks = [task for index, task in enumerate(tasks) if index % shard_count == shard_index]
    failures: list[tuple[str, int]] = []
    for spec, sampler, sample_index in tasks:
        out_dir = combined_output_dir_for(spec, sampler, args, sample_index)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = combined_command_for(spec, sampler, args, sample_index)
        print(
            f"[sample] {spec.run_name}: ckpt={spec.diff_ckpt.name} "
            f"stage1={spec.stage1_ckpt.parent.name} "
            f"scheduler={sampler.scheduler}/{sampler.timesteps} "
            f"sample={'all' if sample_index is None else sample_index} "
            f"decoder={combined_decoder_mode(args)} "
            f"shape={spec.latent_shape} scale={spec.scale_factor}"
        )
        print(" ".join(cmd))
        if args.dry_run:
            continue

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/cardiodit-mpl")
        env.setdefault("XDG_CACHE_HOME", "/tmp/cardiodit-xdg-cache")
        env.setdefault("PYTHONUNBUFFERED", "1")
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            failures.append((
                f"{spec.run_name}/{spec.diff_ckpt.name}/{sampler.label}",
                result.returncode,
            ))

    if failures:
        print("Failed runs:", file=sys.stderr)
        for run_name, status in failures:
            print(f"  - {run_name}: exit {status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import stat
import sys
import tempfile
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import torch
from omegaconf import OmegaConf

from src.scripts.compute_scale_factor import streaming_latent_std
from src.utils.checkpointing import sha256_file


REPO_DIR = Path(__file__).resolve().parents[2]
RUN_ROOT_DEFAULT = Path(
    os.environ.get("MNM2_EVOLUTION_RUN_ROOT", os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs"))
)

RAW_TRAIN_CSV = Path(os.environ.get("MNM2_RAW_TRAIN_CSV", RUN_ROOT_DEFAULT / "data/MNM2/train.csv"))
RAW_VAL_CSV = Path(os.environ.get("MNM2_RAW_VAL_CSV", RUN_ROOT_DEFAULT / "data/MNM2/val.csv"))
FIXED_LATENT_ROOT = Path(
    os.environ.get(
        "MNM2_FIXED_LATENT_ROOT",
        RUN_ROOT_DEFAULT / "data/MNM2_preprocessed/latents/vqgan_ds4_all_dims_wide_e16_quantized_rot2",
    )
)

PYTHON_BIN = sys.executable
TORCHRUN_BIN = str(Path(sys.executable).with_name("torchrun"))

TARGET_UPDATES = 300_000
CHECKPOINT_UPDATES = 10_000
WORLD_SIZE = int(os.environ.get("MNM2_DIT_WORLD_SIZE", "2"))
REMEDIATED_SUFFIX = "_v2_remediated"


def remediated_name(name: str) -> str:
    return f"{name}{REMEDIATED_SUFFIX}"


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _nii_shape(path: str) -> tuple[int, ...]:
    return tuple(int(v) for v in nib.load(path).shape)


def _latent_shape(path: str) -> tuple[int, ...]:
    latent = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(int(v) for v in latent.shape)


def write_raw_splits(run_root: Path) -> dict[str, Path]:
    csv_dir = run_root / "csvs/raw"
    train_rows = _read_csv_rows(RAW_TRAIN_CSV)
    val_rows = _read_csv_rows(RAW_VAL_CSV)

    def enrich(rows: list[dict]) -> list[dict]:
        out = []
        for row in rows:
            shape = _nii_shape(row["image"])
            out.append(
                {
                    "image": row["image"],
                    "T_raw": shape[-1],
                    "D_raw": shape[0],
                    "H_raw": shape[1],
                    "W_raw": shape[2],
                }
            )
        return out

    train_all = enrich(train_rows)
    val_all = enrich(val_rows)
    train_native = [row for row in train_all if int(row["T_raw"]) in (25, 30)]

    paths = {
        "train_all": csv_dir / "train_raw_all.csv",
        "train_native": csv_dir / "train_raw_native_T25_T30.csv",
        "val": csv_dir / "val_raw.csv",
    }
    fields = ["image", "T_raw", "D_raw", "H_raw", "W_raw"]
    _write_csv(paths["train_all"], train_all, fields)
    _write_csv(paths["train_native"], train_native, fields)
    _write_csv(paths["val"], val_all, fields)

    summary = {
        "train_all": len(train_all),
        "train_native_T25_T30": len(train_native),
        "train_excluded_native": [
            row for row in train_all if int(row["T_raw"]) not in (25, 30)
        ],
        "val": len(val_all),
        "train_T_counts": dict(sorted(Counter(int(r["T_raw"]) for r in train_all).items())),
        "val_T_counts": dict(sorted(Counter(int(r["T_raw"]) for r in val_all).items())),
    }
    (csv_dir / "raw_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return paths


def write_fixed_latent_splits(run_root: Path) -> dict[str, Path]:
    out_dir = run_root / "csvs/fixed_latents"
    paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
    }

    for split, src_name in (("train", "train.csv"), ("val", "val.csv")):
        src_csv = FIXED_LATENT_ROOT / src_name
        rows_in = _read_csv_rows(src_csv)
        common_shape = _latent_shape(rows_in[0]["image"]) if rows_in else None
        rows = []
        for row in rows_in:
            shape = common_shape
            rows.append(
                {
                    "image": row["image"],
                    "T_latent": shape[4],
                    "C_latent": shape[0],
                    "Z_latent": shape[1],
                    "X_latent": shape[2],
                    "Y_latent": shape[3],
                }
            )
        _write_csv(
            paths[split],
            rows,
            ["image", "T_latent", "C_latent", "Z_latent", "X_latent", "Y_latent"],
        )

    return paths


HW_ONLY_DS = [[2, 2, 1], [4, 4, 1], 1, [1, 1, 0]]
HW_ONLY_US = [[2, 2, 1], [4, 4, 1], 1, [1, 1, 0], [0, 0, 0]]
ALL_DIMS_DS = [2, 4, 1, 1]
ALL_DIMS_US = [2, 4, 1, 1, 0]


STAGE1_BRANCHES = {
    "S1_ds8xy_noT_native": {
        "train_split": "train_native",
        "downsample": [HW_ONLY_DS, HW_ONLY_DS, HW_ONLY_DS],
        "upsample": [HW_ONLY_US, HW_ONLY_US, HW_ONLY_US],
        "channels": [160, 320, 320],
        "target_frames": "native",
        "time_pad_multiple": None,
        "batch_size": 1,
        "encode_mode": "native",
    },
    "S1_ds4xy_noT_native": {
        "train_split": "train_native",
        "downsample": [HW_ONLY_DS, HW_ONLY_DS],
        "upsample": [HW_ONLY_US, HW_ONLY_US],
        "channels": [160, 320],
        "target_frames": "native",
        "time_pad_multiple": None,
        "batch_size": 1,
        "encode_mode": "native",
    },
    "S1_ds4_all_dims_paddiv": {
        "train_split": "train_all",
        "downsample": [ALL_DIMS_DS, ALL_DIMS_DS],
        "upsample": [ALL_DIMS_US, ALL_DIMS_US],
        "channels": [160, 320],
        "target_frames": "native",
        "time_pad_multiple": 4,
        "batch_size": 1,
        "encode_mode": "paddiv4",
        "disc_last_conv_kernel_size": 1,
    },
    "S1_ds4_all_dims_fixed32": {
        "train_split": "train_all",
        "downsample": [ALL_DIMS_DS, ALL_DIMS_DS],
        "upsample": [ALL_DIMS_US, ALL_DIMS_US],
        "channels": [160, 320],
        "target_frames": 32,
        "time_pad_multiple": None,
        "batch_size": 2,
        "encode_mode": "fixed32",
    },
}


DIT_NEW_FAMILIES = {
    "S1_ds8xy_noT_native": {
        "max_input_size": [12, 28, 28, 30],
        "patch_size": [1, 4, 4, 5],
    },
    "S1_ds4xy_noT_native": {
        "max_input_size": [12, 56, 56, 30],
        "patch_size": [1, 8, 8, 5],
    },
    "S1_ds4_all_dims_paddiv": {
        "max_input_size": [12, 56, 56, 8],
        "patch_size": [1, 8, 8, 1],
    },
}

DIT_SMALL_PATCH_RUNS = {
    "S1_ds8xy_noT_native_ddpm_rope4d_selfcond_patch2x2_t5": {
        "family": "S1_ds8xy_noT_native",
        "max_input_size": [12, 28, 28, 30],
        "patch_size": [1, 2, 2, 5],
        "batch_size": 2,
        "grad_accum_steps": 4,
    },
    "S1_ds4xy_noT_native_ddpm_rope4d_selfcond_patch4x4_t5": {
        "family": "S1_ds4xy_noT_native",
        "max_input_size": [12, 56, 56, 30],
        "patch_size": [1, 4, 4, 5],
        "batch_size": 2,
        "grad_accum_steps": 4,
    },
    "S1_ds4xy_noT_native_ddpm_rope4d_selfcond_patch2x2_t5": {
        "family": "S1_ds4xy_noT_native",
        "max_input_size": [12, 56, 56, 30],
        "patch_size": [1, 2, 2, 5],
        "batch_size": 1,
        "grad_accum_steps": 8,
    },
    "S1_ds4_all_dims_paddiv_ddpm_rope4d_selfcond_patch4x4_t1": {
        "family": "S1_ds4_all_dims_paddiv",
        "max_input_size": [12, 56, 56, 8],
        "patch_size": [1, 4, 4, 1],
        "batch_size": 1,
        "grad_accum_steps": 8,
    },
}


def wandb_cfg() -> dict:
    return {
        "entity": None,
        "project": "CardioDiT-Evolution",
        "offline": True,
        "log_artifacts": False,
    }


def write_stage1_configs(run_root: Path) -> dict[str, Path]:
    cfg_dir = run_root / "configs/stage1"
    paths = {}
    for name, spec in STAGE1_BRANCHES.items():
        run_name = remediated_name(name)
        channels = spec["channels"]
        discriminator_params = {
            "spatial_dims": 3,
            "num_channels": 64,
            "num_layers_d": 3,
            "in_channels": 1,
            "out_channels": 1,
            "norm": "INSTANCE",
        }
        if spec.get("disc_last_conv_kernel_size") is not None:
            discriminator_params["last_conv_kernel_size"] = spec["disc_last_conv_kernel_size"]

        cfg = {
            "model": {
                "params": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_channels": channels,
                    "num_res_channels": channels,
                    "num_res_layers": 3,
                    "downsample_parameters": spec["downsample"],
                    "upsample_parameters": spec["upsample"],
                    "num_embeddings": 8192,
                    "embedding_dim": 16,
                    "commitment_cost": 0.25,
                    "decay": 0.99,
                    "epsilon": 1.0e-5,
                    "ddp_sync": True,
                    "dead_code_threshold": 0.5,
                }
            },
            "discriminator": {
                "params": discriminator_params,
            },
            "training": {
                "n_epochs": 800,
                "eval_freq": 10,
                "batch_size": spec["batch_size"],
                "num_workers": 8,
                "base_lr": 1.0e-5,
                "disc_lr": 3.0e-6,
                "lr_update_gamma": 0.99999,
                "roi_size": [224, 224, 32],
                "target_frames": spec["target_frames"],
                "time_pad_multiple": spec["time_pad_multiple"],
                "use_persistent": True,
                "cache_dir": str(Path("cache") / run_name),
                "spatial_permute": [0, 2, 3, 1],
            },
            "encoding": {
                "target_z": 12,
                "quantized": True,
                "geometry_policy": "require_consistent",
                "phase_endpoint": True,
            },
            "losses": {
                "perceptual_weight": 0.2,
                "jukebox_weight": 0.2,
                "adv_weight": 0.09,
                "adv_warmup": 0,
                "params": {
                    "perceptual_params": {"spatial_dims": 3, "network_type": "squeeze"},
                    "jukebox_params": {"spatial_dims": 3},
                },
            },
            "wandb": wandb_cfg(),
        }
        cfg["run_schema"] = {
            "version": 2,
            "run_name": run_name,
            "legacy_predecessor": name,
        }
        out = cfg_dir / f"{run_name}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(cfg), out)
        paths[run_name] = out
    return paths


def ddpm_scheduler() -> dict:
    return {
        "schedule": "cosine",
        "num_train_timesteps": 1000,
        "prediction_type": "v_prediction",
        "zero_terminal_snr": True,
    }


def flow_scheduler() -> dict:
    return {
        "num_train_timesteps": 1000,
        "sample_method": "logit_normal",
        "logit_mean": 0.5,
        "logit_std": 0.7,
    }


def dit_config(
    *,
    input_size: list[int],
    patch_size: list[int],
    pos_embed_mode: str,
    self_conditioning: bool,
    scheduler_type: str = "ddpm",
    variable_shape: bool = False,
    max_input_size: list[int] | None = None,
    batch_strategy: str = "fixed",
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    preserve_scale_factor: bool = False,
) -> dict:
    model_params = {
        "input_size": input_size,
        "patch_size": patch_size,
        "in_channels": 16,
        "hidden_size": 768,
        "depth": 16,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.0,
        "num_classes": 0,
        "learn_sigma": False,
        "flash_attention": True,
        "attn_drop": 0.0,
        "mlp_drop": 0.0,
        "spacing": None,
        "self_conditioning": self_conditioning,
        "pos_embed_mode": pos_embed_mode,
        "qk_norm": True,
    }
    if variable_shape:
        model_params["variable_shape"] = True
        model_params["max_input_size"] = max_input_size or input_size

    cfg = {
        "model": {"params": model_params},
        "scheduler": flow_scheduler() if scheduler_type == "flow_matching" else ddpm_scheduler(),
        "training": {
            # Production defaults cover the 300k-update plan for the
            # current MNM2 split; prepare-dit resolves exact epochs from the
            # contracted latent manifest before submission.
            "n_epochs": 16000,
            "eval_freq": 526,
            "target_optimizer_updates": TARGET_UPDATES,
            "checkpoint_optimizer_updates": CHECKPOINT_UPDATES,
            "batch_size": batch_size,
            "num_workers": 8,
            "amp_dtype": "bf16",
            "scale_factor": 1.0,
            "preserve_scale_factor": preserve_scale_factor,
            "use_ema": True,
            "ema_decay": 0.9999,
            "ema_warmup_steps": 5000,
            "min_snr_gamma": 0.0,
            "self_conditioning": self_conditioning,
            "offset_noise_strength": 0.1,
            "grad_accum_steps": grad_accum_steps,
            "normalize_latents": True,
            "latent_stats_path": None,
            "phase_dir": None,
            "alpha_pool": "linear",
            "batch_strategy": batch_strategy,
            "preload_latents": True,
            "keep_last_n_checkpoints": -1,
            "objective_loss": "mse",
            "validation_primary_metric": "unconditioned",
            "allow_legacy_latents": False,
            "sample_log": {"enabled": False},
        },
        "optim": {
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "warmup_updates": 5000,
            "total_updates": TARGET_UPDATES,
            "min_lr_ratio": 0.01,
        },
        "wandb": wandb_cfg(),
    }
    if scheduler_type == "flow_matching":
        cfg["scheduler_type"] = "flow_matching"
    return cfg


def write_dit_configs(run_root: Path) -> dict[str, Path]:
    cfg_dir = run_root / "configs/dit"
    paths = {}
    for family, spec in DIT_NEW_FAMILIES.items():
        max_input_size = spec["max_input_size"]
        for suffix, pos_mode in (
            ("ddpm_rope4d_selfcond", "rope4d"),
            ("ddpm_varivit_center_select_selfcond", "center_select"),
        ):
            predecessor = f"{family}_{suffix}"
            run_name = remediated_name(predecessor)
            cfg = dit_config(
                input_size=max_input_size,
                max_input_size=max_input_size,
                patch_size=spec["patch_size"],
                pos_embed_mode=pos_mode,
                self_conditioning=True,
                scheduler_type="ddpm",
                variable_shape=True,
                batch_strategy="bucket",
            )
            cfg["run_schema"] = {
                "version": 2,
                "run_name": run_name,
                "legacy_predecessor": predecessor,
            }
            cfg["latent_preprocessing"] = {
                "stage1_family": remediated_name(family),
                "contract_schema_version": 1,
            }
            out = cfg_dir / "native_padded" / f"{run_name}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(cfg), out)
            paths[run_name] = out

    for run_name, spec in DIT_SMALL_PATCH_RUNS.items():
        max_input_size = spec["max_input_size"]
        tokens = math.prod(
            size // patch
            for size, patch in zip(max_input_size, spec["patch_size"])
        )
        if tokens > 32768:
            # Preserve historical files outside the advertised registry, but
            # never generate a known-infeasible new-run candidate.
            continue
        predecessor = run_name
        run_name = remediated_name(predecessor)
        cfg = dit_config(
            input_size=max_input_size,
            max_input_size=max_input_size,
            patch_size=spec["patch_size"],
            pos_embed_mode="rope4d",
            self_conditioning=True,
            scheduler_type="ddpm",
            variable_shape=True,
            batch_strategy="bucket",
            batch_size=spec["batch_size"],
            grad_accum_steps=spec["grad_accum_steps"],
            preserve_scale_factor=True,
        )
        cfg["run_schema"] = {
            "version": 2,
            "run_name": run_name,
            "legacy_predecessor": predecessor,
        }
        cfg["latent_preprocessing"] = {
            "stage1_family": remediated_name(spec["family"]),
            "contract_schema_version": 1,
        }
        out = cfg_dir / "native_padded" / f"{run_name}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(cfg), out)
        paths[run_name] = out

    return paths


def write_repository_templates(destination: Path) -> list[Path]:
    """Render the sole advertised config/job set into a review tree."""
    if destination.resolve() == REPO_DIR.resolve():
        raise ValueError("Render repository templates to staging, then apply explicitly")
    stage1 = write_stage1_configs(destination)
    dit = write_dit_configs(destination)
    advertised = {
        "schema_version": 1,
        "source_generator": "scripts/mnm2_evolution/prepare_mnm2_evolution.py",
        "stage1": [str(path.relative_to(destination)) for path in stage1.values()],
        "dit": [str(path.relative_to(destination)) for path in dit.values()],
        "legacy_policy": (
            "Other checked-in configs are historical/experimental and require "
            "explicit legacy flags; they are not new-run quick-start configs."
        ),
    }
    registry = destination / "configs/advertised_runs.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(advertised), registry)

    # The HPC module is a rendering backend only; this command is the public
    # source of truth and golden-test entry point.
    from scripts.mnm2_evolution import prepare_hpc_evolution as hpc

    jobs = destination / "jobs"
    job_payloads = {
        "train_dit_evolution.slurm": hpc.train_dit_slurm(),
        "train_vqgan_evolution.slurm": hpc.train_vqgan_slurm(),
        "encode_latents_evolution.slurm": hpc.encode_latents_slurm(),
        "final_manifest_evolution.slurm": hpc.final_manifest_slurm(),
        "remediation_gpu_release_gate.slurm": hpc.remediation_gpu_gate_slurm(),
        "submit_evolution_queue.sh": hpc.submit_script(
            "projects/CardioDiT_MNM2_evolution_20260616", REPO_DIR
        ),
    }
    rendered = [registry, *stage1.values(), *dit.values()]
    for name, text in job_payloads.items():
        path = jobs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if path.suffix == ".sh":
            path.chmod(
                path.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )
        rendered.append(path)
    queue_path = destination / "queue/run_queue.sh"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(queue_script_text(REPO_DIR))
    queue_path.chmod(
        queue_path.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
    rendered.append(queue_path)
    return rendered


def stage_or_apply_repository_templates(staging: Path, apply: bool) -> list[tuple[str, str]]:
    rendered = write_repository_templates(staging)
    changes = []
    for staged in rendered:
        relative = staged.relative_to(staging)
        current = REPO_DIR / relative
        if not current.is_file():
            status = "ADD"
        elif current.read_bytes() != staged.read_bytes():
            status = "CHANGE"
        elif bool(current.stat().st_mode & stat.S_IXUSR) != bool(
            staged.stat().st_mode & stat.S_IXUSR
        ):
            status = "MODE"
        else:
            status = "SAME"
        changes.append((status, str(relative)))
        if apply and status != "SAME":
            current.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, current)
    return changes


def _count_rows(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _shape_counts(path: Path) -> Counter:
    rows = _read_csv_rows(path)
    counts = Counter()
    for row in rows:
        if all(k in row and row[k] not in ("", None) for k in ("C_latent", "Z_latent", "X_latent", "Y_latent", "T_latent")):
            key = tuple(int(row[k]) for k in ("C_latent", "Z_latent", "X_latent", "Y_latent", "T_latent"))
        else:
            key = _latent_shape(row["image"])
        counts[key] += 1
    return counts


def batches_per_epoch(train_csv: Path, batch_size: int, world_size: int, batch_strategy: str) -> int:
    if batch_strategy == "bucket":
        global_batch = batch_size * world_size
        return sum(math.ceil(count / global_batch) for count in _shape_counts(train_csv).values())
    n = _count_rows(train_csv)
    samples_per_rank = math.ceil(n / world_size)
    return math.ceil(samples_per_rank / batch_size)


def compute_scale(train_csv: Path) -> tuple[float, float, int]:
    rows = _read_csv_rows(train_csv)
    paths = [row["image"] for row in rows]
    std = streaming_latent_std(paths)
    return 1.0 / std, std, len(paths)


def prepare_dit_config(
    config_path: Path,
    train_csv: Path,
    scale_out: Path,
    output_config: Path,
    target_updates: int = TARGET_UPDATES,
    checkpoint_updates: int = CHECKPOINT_UPDATES,
    world_size: int = WORLD_SIZE,
) -> dict:
    config_path = Path(config_path).resolve()
    output_config = Path(output_config).resolve()
    if output_config == config_path:
        raise ValueError("Source and resolved DiT config paths must differ")
    cfg = OmegaConf.load(config_path)
    batch_size = int(cfg.training.batch_size)
    grad_accum_steps = int(cfg.training.get("grad_accum_steps", 1))
    batch_strategy = str(cfg.training.get("batch_strategy", "fixed")).lower()
    batches = batches_per_epoch(train_csv, batch_size, world_size, batch_strategy)
    updates = max(1, math.ceil(batches / grad_accum_steps))
    n_epochs = max(1, math.ceil(target_updates / updates))
    eval_freq = max(1, round(checkpoint_updates / updates))
    computed_scale_factor, std, n_latents = compute_scale(train_csv)
    preserve_scale_factor = bool(cfg.training.get("preserve_scale_factor", False))
    configured_scale_factor = float(cfg.training.get("scale_factor", 1.0))
    normalize_latents = bool(cfg.training.get("normalize_latents", False))
    scale_factor = (
        1.0
        if normalize_latents
        else (configured_scale_factor if preserve_scale_factor else computed_scale_factor)
    )

    cfg.training.n_epochs = int(n_epochs)
    cfg.training.eval_freq = int(eval_freq)
    cfg.training.target_optimizer_updates = int(target_updates)
    cfg.training.checkpoint_optimizer_updates = int(checkpoint_updates)
    cfg.training.scale_factor = float(scale_factor)
    cfg.training.keep_last_n_checkpoints = -1
    cfg.training.sample_log = {"enabled": False}
    if "optim" in cfg:
        cfg.optim.total_updates = int(target_updates)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_config)

    payload = {
        "source_config": str(config_path),
        "resolved_config": str(output_config),
        "train_csv": str(train_csv),
        "n_latents": n_latents,
        "batch_size_per_gpu": batch_size,
        "world_size": world_size,
        "batch_strategy": batch_strategy,
        "grad_accum_steps": grad_accum_steps,
        "batches_per_epoch_per_rank": batches,
        "optimizer_updates_per_epoch": updates,
        "target_optimizer_updates": target_updates,
        "checkpoint_optimizer_updates": checkpoint_updates,
        "n_epochs": n_epochs,
        "eval_freq": eval_freq,
        "std": std,
        "computed_scale_factor": computed_scale_factor,
        "preserve_scale_factor": preserve_scale_factor,
        "scale_factor": scale_factor,
        "scale_policy": (
            "per_channel_normalization_unit_global_scale"
            if normalize_latents
            else "raw_latents_global_scale"
        ),
    }
    scale_out.parent.mkdir(parents=True, exist_ok=True)
    scale_out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def compute_scale_file(train_csv: Path, scale_out: Path) -> dict:
    scale_factor, std, n_latents = compute_scale(train_csv)
    payload = {
        "train_csv": str(train_csv),
        "n_latents": n_latents,
        "std": std,
        "scale_factor": scale_factor,
    }
    scale_out.parent.mkdir(parents=True, exist_ok=True)
    scale_out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def planned_manifest(run_root: Path, stage1_cfgs: dict[str, Path], dit_cfgs: dict[str, Path]) -> list[dict]:
    rows = []
    for name, cfg_path in stage1_cfgs.items():
        predecessor = name.removesuffix(REMEDIATED_SUFFIX)
        spec = STAGE1_BRANCHES[predecessor]
        rows.append(
            {
                "kind": "stage1_vqgan",
                "run_name": name,
                "config_path": str(cfg_path),
                "output_path": str(run_root / "outputs/stage1" / name),
                "wandb_run_name": name,
                "train_csv": str(run_root / "csvs/raw" / ("train_raw_native_T25_T30.csv" if spec["train_split"] == "train_native" else "train_raw_all.csv")),
                "val_csv": str(run_root / "csvs/raw/val_raw.csv"),
                "latent_source": "",
                "stage1_dependency": "",
            }
        )

    fixed_train = run_root / "csvs/fixed_latents/train.csv"
    fixed_val = run_root / "csvs/fixed_latents/val.csv"
    for run_name, cfg_path in dit_cfgs.items():
        family = next(name for name in DIT_NEW_FAMILIES if run_name.startswith(name))
        stage1_dep = remediated_name(family)
        train_csv = run_root / "latents" / stage1_dep / "train/latents.csv"
        val_csv = run_root / "latents" / stage1_dep / "val/latents.csv"
        latent_source = str(run_root / "latents" / stage1_dep)
        rows.append(
            {
                "kind": "stage2_dit",
                "run_name": run_name,
                "config_path": str(cfg_path),
                "output_path": str(run_root / "outputs/dit" / run_name),
                "wandb_run_name": run_name,
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "latent_source": latent_source,
                "stage1_dependency": stage1_dep,
            }
        )
    return rows


def write_plan_manifest(run_root: Path, stage1_cfgs: dict[str, Path], dit_cfgs: dict[str, Path]) -> None:
    rows = planned_manifest(run_root, stage1_cfgs, dit_cfgs)
    manifest_path = run_root / "manifest_plan.csv"
    _write_csv(
        manifest_path,
        rows,
        [
            "kind",
            "run_name",
            "config_path",
            "output_path",
            "wandb_run_name",
            "train_csv",
            "val_csv",
            "latent_source",
            "stage1_dependency",
        ],
    )
    (run_root / "manifest_plan.json").write_text(json.dumps(rows, indent=2) + "\n")


def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _file_hash(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _pointer_target(output_path: Path) -> Path | None:
    pointer = output_path / "last_checkpoint.json"
    if not pointer.is_file():
        return None
    try:
        target = output_path / json.loads(pointer.read_text())["target"]
    except (OSError, ValueError, KeyError):
        return None
    return target if target.is_file() else None


def discover_run_records(run_root: Path) -> list[dict]:
    """Discover real run directories rather than treating a plan as results."""
    plan_path = run_root / "manifest_plan.csv"
    planned = {
        row["run_name"]: row for row in _read_csv_rows(plan_path)
    } if plan_path.is_file() else {}
    records = []
    for kind, outputs in (
        ("stage1_vqgan", run_root / "outputs/stage1"),
        ("stage2_dit", run_root / "outputs/dit"),
    ):
        if not outputs.is_dir():
            continue
        for output_path in sorted(path for path in outputs.iterdir() if path.is_dir()):
            row = dict(planned.get(output_path.name, {}))
            periodic_epochs = sorted(
                output_path.glob("checkpoint_epoch_*.pth"), key=_checkpoint_epoch
            )
            periodic_updates = sorted(
                output_path.glob("checkpoint_update_*.pth"), key=_checkpoint_epoch
            )
            periodic = [*periodic_epochs, *periodic_updates]
            final = output_path / "final_model.pth"
            best = output_path / "best_model.pth"
            canonical_last = _pointer_target(output_path)
            legacy_last = output_path / "last_checkpoint.pth"
            latest = canonical_last or (periodic[-1] if periodic else None)
            if latest is None and final.is_file():
                latest = final
            if latest is None and best.is_file():
                latest = best
            if latest is None and legacy_last.is_file():
                latest = legacy_last
            config_path = Path(row.get("config_path", ""))
            train_csv = Path(row.get("train_csv", ""))
            val_csv = Path(row.get("val_csv", ""))
            row.update({
                "manifest_schema_version": 1,
                "kind": kind,
                "run_name": output_path.name,
                "output_path": str(output_path),
                "status": "complete" if final.is_file() else ("checkpointed" if latest else "started"),
                "config_sha256": _file_hash(config_path),
                "train_manifest_sha256": _file_hash(train_csv),
                "validation_manifest_sha256": _file_hash(val_csv),
                "best_checkpoint": str(best) if best.is_file() else "",
                "last_checkpoint_pointer": str(output_path / "last_checkpoint.json") if canonical_last else "",
                "legacy_last_checkpoint": str(legacy_last) if legacy_last.is_file() else "",
                "final_model": str(final) if final.is_file() else "",
                "periodic_checkpoints": json.dumps([str(path) for path in periodic]),
                "periodic_epochs": json.dumps(
                    [_checkpoint_epoch(path) for path in periodic_epochs]
                ),
                "periodic_updates": json.dumps(
                    [_checkpoint_epoch(path) for path in periodic_updates]
                ),
                "latest_checkpoint_sha256": _file_hash(latest) if latest else "",
            })
            records.append(row)
    return records


def write_final_manifest(run_root: Path) -> Path:
    final_rows = discover_run_records(run_root)
    fields = sorted({key for row in final_rows for key in row}) or ["manifest_schema_version"]
    out = run_root / "final_manifest.csv"
    _write_csv(out, final_rows, fields)
    (run_root / "final_manifest.json").write_text(json.dumps(final_rows, indent=2) + "\n")
    return out


def queue_script_text(run_root: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'RUN_ROOT="{run_root}"',
        f'REPO_DIR="{REPO_DIR}"',
        f'PYTHON="{PYTHON_BIN}"',
        f'TORCHRUN="{TORCHRUN_BIN}"',
        'PREP="${REPO_DIR}/scripts/mnm2_evolution/prepare_mnm2_evolution.py"',
        'LOG_DIR="${RUN_ROOT}/logs"',
        'mkdir -p "${LOG_DIR}" "${RUN_ROOT}/outputs/stage1" "${RUN_ROOT}/outputs/dit"',
        'QUEUE_LOG="${LOG_DIR}/run_queue.log"',
        "",
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"',
        'export MNM2_DIT_WORLD_SIZE="${MNM2_DIT_WORLD_SIZE:-2}"',
        'export WANDB_MODE="online"',
        'export WANDB_ENTITY="marvins-aicm"',
        'export WANDB_PROJECT="CardioDiT-Evolution"',
        'export PYTHONUNBUFFERED=1',
        'export NCCL_ASYNC_ERROR_HANDLING=1',
        'export TORCH_NCCL_ASYNC_ERROR_HANDLING=1',
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        'export MPLCONFIGDIR="/tmp/matplotlib-cardiodit-mnm2-evolution"',
        "",
        "log_msg() {",
        "  printf '[%s] %s\\n' \"$(date --iso-8601=seconds)\" \"$*\" | tee -a \"${QUEUE_LOG}\"",
        "}",
        "",
        "run_logged() {",
        "  local log_file=\"$1\"",
        "  shift",
        "  log_msg \"Running: $*\"",
        "  \"$@\" >> \"${log_file}\" 2>&1",
        "}",
        "",
        "wait_for_gpus() {",
        "  while true; do",
        "    local busy",
        "    busy=\"$(nvidia-smi --id=0,1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)\"",
        "    if [[ -z \"${busy}\" ]]; then",
        "      log_msg \"GPUs 0,1 are idle.\"",
        "      return 0",
        "    fi",
        "    log_msg \"Waiting for GPUs 0,1 to be idle. Active compute processes: ${busy}\"",
        "    sleep 300",
        "  done",
        "}",
        "",
        "run_tests() {",
        '  log_msg "Running focused unit/smoke tests."',
        '  cd "${REPO_DIR}"',
        '  run_logged "${LOG_DIR}/tests.log" "${PYTHON}" -m pytest \\',
        "    tests/test_dit_trainer_logging.py \\",
        "    tests/test_dataloading_buckets.py \\",
        "    tests/test_encode_latents.py \\",
        "    tests/test_train_vqgan_config.py \\",
        "    -q",
        "}",
        "",
        "generate_artifacts() {",
        '  log_msg "Generating external configs, split CSVs, and plan manifest."',
        '  run_logged "${LOG_DIR}/generate.log" "${PYTHON}" "${PREP}" generate --run-root "${RUN_ROOT}" --no-queue --apply',
        "}",
        "",
        "prepare_dit() {",
        "  local run_name=\"$1\"",
        "  local config=\"$2\"",
        "  local train_csv=\"$3\"",
        '  run_logged "${LOG_DIR}/${run_name}_prepare.log" "${PYTHON}" "${PREP}" prepare-dit \\',
        '    --config "${config}" \\',
        '    --output-config "${RUN_ROOT}/resolved_configs/dit/${run_name}.yaml" \\',
        '    --train-csv "${train_csv}" \\',
        '    --world-size "${MNM2_DIT_WORLD_SIZE}" \\',
        '    --scale-out "${RUN_ROOT}/scale_factors/${run_name}.json"',
        "}",
        "",
        "compute_family_scale() {",
        "  local family=\"$1\"",
        "  local train_csv=\"$2\"",
        '  run_logged "${LOG_DIR}/${family}_scale.log" "${PYTHON}" "${PREP}" compute-scale \\',
        '    --train-csv "${train_csv}" \\',
        '    --scale-out "${RUN_ROOT}/scale_factors/${family}.json"',
        "}",
        "",
        "run_dit() {",
        "  local run_name=\"$1\"",
        "  local source_config=\"$2\"",
        '  local config="${RUN_ROOT}/resolved_configs/dit/${run_name}.yaml"',
        "  local train_csv=\"$3\"",
        "  local val_csv=\"$4\"",
        '  local run_dir="${RUN_ROOT}/outputs/dit/${run_name}"',
        '  if [[ -f "${run_dir}/final_model.pth" ]]; then',
        '    log_msg "Skipping ${run_name}: final_model.pth already exists."',
        "    return 0",
        "  fi",
        "  wait_for_gpus",
        '  log_msg "Starting DiT ${run_name}."',
        '  cd "${REPO_DIR}"',
        '  run_logged "${LOG_DIR}/${run_name}.log" "${TORCHRUN}" --standalone --nproc_per_node="${MNM2_DIT_WORLD_SIZE}" \\',
        '    "${REPO_DIR}/src/scripts/train_dit.py" \\',
        '    --config "${config}" \\',
        '    --output_dir "${RUN_ROOT}/outputs/dit" \\',
        '    --run_name "${run_name}" \\',
        '    --training_ids "${train_csv}" \\',
        '    --validation_ids "${val_csv}"',
        "}",
        "",
        "run_vqgan() {",
        "  local family=\"$1\"",
        "  local config=\"$2\"",
        "  local train_csv=\"$3\"",
        "  local val_csv=\"$4\"",
        '  local run_dir="${RUN_ROOT}/outputs/stage1/${family}"',
        '  if [[ -f "${run_dir}/final_model.pth" ]]; then',
        '    log_msg "Skipping Stage 1 ${family}: final_model.pth already exists."',
        "    return 0",
        "  fi",
        "  wait_for_gpus",
        '  log_msg "Starting Stage 1 ${family}."',
        '  cd "${REPO_DIR}"',
        '  run_logged "${LOG_DIR}/${family}.log" "${TORCHRUN}" --standalone --nproc_per_node=2 \\',
        '    "${REPO_DIR}/src/scripts/train_vqgan.py" \\',
        '    --cache_dir "${RUN_ROOT}/cache/${family}" \\',
        '    --config "${config}" \\',
        '    --output_dir "${RUN_ROOT}/outputs/stage1" \\',
        '    --run_name "${family}" \\',
        '    --training_ids "${train_csv}" \\',
        '    --validation_ids "${val_csv}"',
        "}",
        "",
        "encode_branch() {",
        "  local family=\"$1\"",
        "  local config=\"$2\"",
        "  local train_csv=\"$3\"",
        "  local val_csv=\"$4\"",
        "  local mode=\"$5\"",
        '  local stage1_dir="${RUN_ROOT}/outputs/stage1/${family}"',
        '  local ckpt=""',
        '  if [[ -f "${stage1_dir}/last_checkpoint.json" ]]; then',
        '    ckpt="$("${PYTHON}" -c \'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["target"])\' "${stage1_dir}/last_checkpoint.json")"',
        '  elif [[ -f "${stage1_dir}/best_model.pth" ]]; then ckpt="${stage1_dir}/best_model.pth";',
        '  elif [[ -f "${stage1_dir}/final_model.pth" ]]; then ckpt="${stage1_dir}/final_model.pth";',
        '  elif [[ -f "${stage1_dir}/last_checkpoint.pth" ]]; then ckpt="${stage1_dir}/last_checkpoint.pth"; fi',
        '  if [[ ! -f "${ckpt}" ]]; then',
        '    log_msg "No checkpoint found for ${family}."',
        "    return 1",
        "  fi",
        "  local target_args=()",
        '  if [[ "${mode}" == "native" ]]; then',
        '    target_args=(--target_frames native)',
        '  elif [[ "${mode}" == "paddiv4" ]]; then',
        '    target_args=(--target_frame_multiple 4)',
        "  else",
        '    target_args=(--target_frames 32)',
        "  fi",
        "  for split in train val; do",
        "    local csv_path",
        '    if [[ "${split}" == "train" ]]; then csv_path="${train_csv}"; else csv_path="${val_csv}"; fi',
        '    local out_dir="${RUN_ROOT}/latents/${family}/${split}"',
        "    wait_for_gpus",
        '    log_msg "Encoding latents ${family}/${split}."',
        '    cd "${REPO_DIR}"',
        '    run_logged "${LOG_DIR}/${family}_encode_${split}.log" "${PYTHON}" "${REPO_DIR}/src/scripts/encode_latents.py" \\',
        '      --csv "${csv_path}" \\',
        '      --output_dir "${out_dir}" \\',
        '      --vqvae_ckpt "${ckpt}" \\',
        '      --config "${config}" \\',
        "      --roi_size 224 224 32 \\",
        "      --target_z 12 \\",
        "      --device cuda:0 \\",
        "      --batch_size 4 \\",
        "      --dim_perm 2 3 1 0 \\",
        "      --quantized \\",
        "      --geometry_policy require_consistent \\",
        '      "${target_args[@]}"',
        "  done",
        "}",
        "",
        "run_tests",
        "generate_artifacts",
        "",
        'RAW_TRAIN_ALL="${RUN_ROOT}/csvs/raw/train_raw_all.csv"',
        'RAW_TRAIN_NATIVE="${RUN_ROOT}/csvs/raw/train_raw_native_T25_T30.csv"',
        'RAW_VAL="${RUN_ROOT}/csvs/raw/val_raw.csv"',
        "",
        "run_vqgan S1_ds8xy_noT_native_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds8xy_noT_native_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_NATIVE}" "${RAW_VAL}"',
        "encode_branch S1_ds8xy_noT_native_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds8xy_noT_native_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_NATIVE}" "${RAW_VAL}" native',
        'compute_family_scale S1_ds8xy_noT_native_v2_remediated "${RUN_ROOT}/latents/S1_ds8xy_noT_native_v2_remediated/train/latents.csv"',
        "",
        "run_vqgan S1_ds4xy_noT_native_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds4xy_noT_native_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_NATIVE}" "${RAW_VAL}"',
        "encode_branch S1_ds4xy_noT_native_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds4xy_noT_native_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_NATIVE}" "${RAW_VAL}" native',
        'compute_family_scale S1_ds4xy_noT_native_v2_remediated "${RUN_ROOT}/latents/S1_ds4xy_noT_native_v2_remediated/train/latents.csv"',
        "",
        "run_vqgan S1_ds4_all_dims_paddiv_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds4_all_dims_paddiv_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_ALL}" "${RAW_VAL}"',
        "encode_branch S1_ds4_all_dims_paddiv_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds4_all_dims_paddiv_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_ALL}" "${RAW_VAL}" paddiv4',
        'compute_family_scale S1_ds4_all_dims_paddiv_v2_remediated "${RUN_ROOT}/latents/S1_ds4_all_dims_paddiv_v2_remediated/train/latents.csv"',
        "",
        "run_vqgan S1_ds4_all_dims_fixed32_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds4_all_dims_fixed32_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_ALL}" "${RAW_VAL}"',
        "encode_branch S1_ds4_all_dims_fixed32_v2_remediated \\",
        '  "${RUN_ROOT}/configs/stage1/S1_ds4_all_dims_fixed32_v2_remediated.yaml" \\',
        '  "${RAW_TRAIN_ALL}" "${RAW_VAL}" fixed32',
        'compute_family_scale S1_ds4_all_dims_fixed32_v2_remediated "${RUN_ROOT}/latents/S1_ds4_all_dims_fixed32_v2_remediated/train/latents.csv"',
        "",
        "for family in S1_ds8xy_noT_native_v2_remediated S1_ds4xy_noT_native_v2_remediated S1_ds4_all_dims_paddiv_v2_remediated; do",
        "  for suffix in ddpm_rope4d_selfcond ddpm_varivit_center_select_selfcond; do",
        '    legacy_family="${family%_v2_remediated}"',
        '    run_name="${legacy_family}_${suffix}_v2_remediated"',
        '    config="${RUN_ROOT}/configs/dit/native_padded/${run_name}.yaml"',
        '    train_csv="${RUN_ROOT}/latents/${family}/train/latents.csv"',
        '    val_csv="${RUN_ROOT}/latents/${family}/val/latents.csv"',
        '    prepare_dit "${run_name}" "${config}" "${train_csv}"',
        '    run_dit "${run_name}" "${config}" "${train_csv}" "${val_csv}"',
        "  done",
        "done",
        "",
        "for run_name in \\",
        "  S1_ds8xy_noT_native_ddpm_rope4d_selfcond_patch2x2_t5_v2_remediated \\",
        "  S1_ds4xy_noT_native_ddpm_rope4d_selfcond_patch4x4_t5_v2_remediated \\",
        "  S1_ds4_all_dims_paddiv_ddpm_rope4d_selfcond_patch4x4_t1_v2_remediated; do",
        '  family="${run_name%%_ddpm_rope4d_selfcond*}"',
        '  stage1_family="${family}_v2_remediated"',
        '  config="${RUN_ROOT}/configs/dit/native_padded/${run_name}.yaml"',
        '  train_csv="${RUN_ROOT}/latents/${stage1_family}/train/latents.csv"',
        '  val_csv="${RUN_ROOT}/latents/${stage1_family}/val/latents.csv"',
        '  prepare_dit "${run_name}" "${config}" "${train_csv}"',
        '  run_dit "${run_name}" "${config}" "${train_csv}" "${val_csv}"',
        "done",
        "",
        'run_logged "${LOG_DIR}/final_manifest.log" "${PYTHON}" "${PREP}" final-manifest --run-root "${RUN_ROOT}"',
        'log_msg "MNM2 evolution queue completed."',
        "",
    ]
    return "\n".join(lines)


def generate(run_root: Path, write_queue: bool = True) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for subdir in (
        "configs",
        "csvs",
        "latents",
        "logs",
        "outputs/stage1",
        "outputs/dit",
        "queue",
        "scale_factors",
        "cache",
    ):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    write_raw_splits(run_root)
    write_fixed_latent_splits(run_root)
    stage1_cfgs = write_stage1_configs(run_root)
    dit_cfgs = write_dit_configs(run_root)
    write_plan_manifest(run_root, stage1_cfgs, dit_cfgs)

    metadata = {
        "run_root": str(run_root),
        "repo_dir": str(REPO_DIR),
        "target_optimizer_updates": TARGET_UPDATES,
        "checkpoint_optimizer_updates": CHECKPOINT_UPDATES,
        "world_size": WORLD_SIZE,
        "wandb": {
            "mode": "online",
            "entity": None,
            "project": "CardioDiT-Evolution",
            "log_artifacts": False,
        },
        "fixed_latent_root": str(FIXED_LATENT_ROOT),
    }
    (run_root / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if write_queue:
        queue_path = run_root / "queue/run_queue.sh"
        queue_path.write_text(queue_script_text(run_root))
        mode = queue_path.stat().st_mode
        queue_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    gen.add_argument("--no-queue", action="store_true")
    gen.add_argument(
        "--apply", action="store_true",
        help="Write to --run-root; without this flag generate a review staging tree.",
    )
    gen.add_argument("--staging-root", type=Path, default=None)

    prep = sub.add_parser("prepare-dit")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output-config", type=Path, required=True)
    prep.add_argument("--train-csv", type=Path, required=True)
    prep.add_argument("--scale-out", type=Path, required=True)
    prep.add_argument("--target-updates", type=int, default=TARGET_UPDATES)
    prep.add_argument("--checkpoint-updates", type=int, default=CHECKPOINT_UPDATES)
    prep.add_argument("--world-size", type=int, default=WORLD_SIZE)

    scale = sub.add_parser("compute-scale")
    scale.add_argument("--train-csv", type=Path, required=True)
    scale.add_argument("--scale-out", type=Path, required=True)

    final = sub.add_parser("final-manifest")
    final.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)

    templates = sub.add_parser("repo-templates")
    templates.add_argument("--apply", action="store_true")
    templates.add_argument("--staging-root", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        destination = args.run_root if args.apply else (
            args.staging_root
            or Path(tempfile.mkdtemp(prefix="cardiodit-run-staging-"))
        )
        if destination.resolve() == REPO_DIR.resolve():
            raise ValueError("Refusing to generate a run tree over the source repository")
        generate(destination, write_queue=not args.no_queue)
        if not args.apply:
            print(f"Staged generation at {destination}")
            for staged in sorted(path for path in destination.rglob("*") if path.is_file()):
                relative = staged.relative_to(destination)
                current = args.run_root / relative
                status = "ADD" if not current.is_file() else (
                    "SAME" if staged.read_bytes() == current.read_bytes() else "CHANGE"
                )
                print(f"{status:6} {relative}")
            print("Review the staging tree, then rerun with --apply to publish it.")
    elif args.command == "prepare-dit":
        payload = prepare_dit_config(
            args.config,
            args.train_csv,
            args.scale_out,
            args.output_config,
            target_updates=args.target_updates,
            checkpoint_updates=args.checkpoint_updates,
            world_size=args.world_size,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "compute-scale":
        payload = compute_scale_file(args.train_csv, args.scale_out)
        print(json.dumps(payload, indent=2))
    elif args.command == "final-manifest":
        print(write_final_manifest(args.run_root))
    elif args.command == "repo-templates":
        staging = args.staging_root or Path(
            tempfile.mkdtemp(prefix="cardiodit-repo-templates-")
        )
        if staging.resolve() == REPO_DIR.resolve():
            raise ValueError("staging root must differ from repository root")
        changes = stage_or_apply_repository_templates(staging, args.apply)
        for status, relative in changes:
            print(f"{status:6} {relative}")
        print("Applied." if args.apply else f"Review staging at {staging}; rerun with --apply.")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

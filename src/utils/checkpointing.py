"""Versioned, atomic training checkpoints and reproducibility metadata.

The module is intentionally trainer-agnostic.  DiT and VQ-GAN can share the
same on-disk contract without sharing model-specific restore logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_NAME = "cardiodit.training-checkpoint"
SCHEMA_VERSION = 1
POINTER_NAME = "last_checkpoint.json"
METADATA_NAME = "run_metadata.json"


class CheckpointError(RuntimeError):
    """Base class for checkpoint contract errors."""


class CheckpointMismatchError(CheckpointError):
    """Raised when the current run identities do not match a checkpoint."""


def _jsonable(value: Any) -> Any:
    """Convert config/provenance values to a canonical JSON representation."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolved_config_resume_hash(value: Mapping[str, Any]) -> str:
    """Hash config fields that must remain identical across a strict resume."""
    config = _jsonable(value)
    training = config.get("training") if isinstance(config, dict) else None
    if isinstance(training, dict):
        # Validation/checkpoint cadence is operational. Validation runs under
        # isolated RNG state, so changing these fields does not change updates.
        training.pop("eval_freq", None)
        training.pop("checkpoint_optimizer_updates", None)
        # Accepted for checkpoints produced by the brief 10k metric-logging
        # configuration before checkpoint cadence was clarified.
        training.pop("log_optimizer_updates", None)
    return canonical_hash(config)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json_save(payload: Mapping[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_torch_save(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        with open(tmp_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_hash_cache(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / METADATA_NAME
    if not path.is_file():
        return {"version": 1, "file_hashes": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "file_hashes": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("file_hashes"), dict):
        return {"version": 1, "file_hashes": {}}
    return payload


def cached_file_identity(path: str | Path | None, cache: dict[str, Any]) -> dict[str, Any] | None:
    """Hash a file once per (resolved path, size, mtime), updating ``cache``."""
    if path in (None, ""):
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Provenance input does not exist: {resolved}")
    stat = resolved.stat()
    key = str(resolved)
    old = cache.setdefault("file_hashes", {}).get(key)
    if (
        isinstance(old, dict)
        and old.get("size") == stat.st_size
        and old.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(old.get("sha256"), str)
    ):
        return dict(old)
    identity = {
        "path": key,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }
    cache["file_hashes"][key] = identity
    return dict(identity)


def build_provenance(
    *,
    resolved_config: Mapping[str, Any],
    train_manifest: str | Path,
    validation_manifest: str | Path,
    latent_preprocessing: Mapping[str, Any],
    run_dir: str | Path,
    stage1_checkpoint: str | Path | None = None,
    stage1_config: str | Path | None = None,
    attention_estimate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical run identities and persist the expensive file-hash cache."""
    cache = load_hash_cache(run_dir)
    provenance = {
        "resolved_config": {
            "sha256": canonical_hash(resolved_config),
            "resume_sha256": resolved_config_resume_hash(resolved_config),
        },
        "train_manifest": cached_file_identity(train_manifest, cache),
        "validation_manifest": cached_file_identity(validation_manifest, cache),
        "stage1_checkpoint": cached_file_identity(stage1_checkpoint, cache),
        "stage1_config": cached_file_identity(stage1_config, cache),
        "latent_preprocessing": {
            "sha256": canonical_hash(latent_preprocessing),
            "contract": _jsonable(latent_preprocessing),
        },
        "attention_estimate": (
            {
                "sha256": canonical_hash(attention_estimate),
                "estimate": _jsonable(attention_estimate),
            }
            if attention_estimate is not None
            else None
        ),
    }
    cache["provenance"] = provenance
    atomic_json_save(cache, Path(run_dir) / METADATA_NAME)
    return provenance


def capture_local_rng_state(rank: int | None = None) -> dict[str, Any]:
    if rank is None:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
    return {
        "rank": int(rank),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        # CUDA state is rank-local: do not use get_rng_state_all here.
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def gather_rng_states() -> dict[str, Any]:
    local = capture_local_rng_state()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
    else:
        world_size = 1
        gathered = [local]
    gathered.sort(key=lambda item: item["rank"])
    return {"world_size": world_size, "by_rank": gathered}


def restore_rng_state(bundle: Mapping[str, Any], rank: int | None = None) -> None:
    if rank is None:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
    states = bundle.get("by_rank")
    if not isinstance(states, list):
        raise CheckpointError("Checkpoint RNG state has no rank-indexed 'by_rank' list")
    state = next((item for item in states if item.get("rank") == rank), None)
    if state is None:
        raise CheckpointError(
            f"Checkpoint has no RNG state for rank {rank}; available ranks are "
            f"{[item.get('rank') for item in states]}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])


_LEGACY_FIELDS = (
    "model",
    "ema",
    "optimizer",
    "lr_scheduler",
    "scaler",
    "epoch",
    "best_loss",
    "latent_mean",
    "latent_std",
    "scale_factor",
    "normalize_latents",
    "rng",
    "discriminator",
    "optimizer_g",
    "optimizer_d",
    "scheduler_g",
    "scheduler_d",
    "scaler_g",
    "scaler_d",
    "ema_cluster_size",
    "ema_w",
)


def adapt_legacy_checkpoint(payload: Any) -> dict[str, Any]:
    """Read-only adapter that makes absence explicit and never fabricates state."""
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint payload must be a mapping")
    payload = dict(payload)
    if payload.get("checkpoint_schema", {}).get("name") == SCHEMA_NAME:
        return payload

    # A plain state dict is raw inference state, not a resumable training state.
    looks_like_state_dict = bool(payload) and all(
        isinstance(key, str) and isinstance(value, (torch.Tensor, torch.nn.Parameter))
        for key, value in payload.items()
    )
    source = {"model": payload} if looks_like_state_dict else payload
    available = {field: field in source and source[field] is not None for field in _LEGACY_FIELDS}
    adapted = {field: source.get(field) for field in _LEGACY_FIELDS}
    adapted.update(
        {
            "checkpoint_schema": {"name": "legacy", "version": 0},
            "availability": available,
            "progress": None,
            "metrics": None,
            "resolved_config": None,
            "run_id": None,
            "provenance": None,
            "resume_overrides": [],
            "legacy": True,
        }
    )
    return adapted


def prepare_model_state_for_load(
    checkpoint: Mapping[str, Any],
    current_state_keys: set[str],
    *,
    allow_legacy_rope4d_pos_embed: bool = False,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Apply the sole supported legacy model-state migration.

    New-schema states are returned untouched and therefore remain strict.  An
    old rope4d checkpoint may contain the formerly allocated fixed
    ``pos_embed`` tensor; current rope4d models intentionally have no such key.
    """
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise CheckpointError("Checkpoint model state is unavailable or invalid")
    if not checkpoint.get("legacy") or not allow_legacy_rope4d_pos_embed:
        return state, []
    if "pos_embed" not in state or "pos_embed" in current_state_keys:
        return state, []
    migrated = dict(state)
    removed = migrated.pop("pos_embed")
    shape = list(removed.shape) if isinstance(removed, torch.Tensor) else None
    return migrated, [
        {
            "migration": "legacy_rope4d_pos_embed_removed",
            "field": "model.pos_embed",
            "shape": shape,
        }
    ]


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_provenance: Mapping[str, Any] | None = None,
    *,
    allow_mismatch: bool = False,
) -> list[dict[str, Any]]:
    schema = checkpoint.get("checkpoint_schema")
    if not isinstance(schema, Mapping):
        raise CheckpointError("Checkpoint has no schema descriptor")
    if schema.get("name") == "legacy":
        if expected_provenance is None:
            return []
        mismatches = [
            {"field": field, "checkpoint": None, "current": value.get("sha256")}
            for field, value in expected_provenance.items()
            if isinstance(value, Mapping) and value.get("sha256") is not None
        ]
        if mismatches and not allow_mismatch:
            raise CheckpointMismatchError(
                "Legacy checkpoint has no provenance to validate. Pass the explicit "
                "mismatch override only after independently verifying its inputs."
            )
        return mismatches
    if schema.get("name") != SCHEMA_NAME or schema.get("version") != SCHEMA_VERSION:
        raise CheckpointError(f"Unsupported checkpoint schema: {schema!r}")

    required = ("model", "progress", "resolved_config", "run_id", "provenance", "availability")
    missing = [field for field in required if field not in checkpoint]
    if missing:
        raise CheckpointError(f"Incomplete checkpoint; missing required fields: {missing}")
    if checkpoint["model"] is None:
        raise CheckpointError("Incomplete checkpoint; raw model state is unavailable")
    if checkpoint["run_id"] in (None, ""):
        raise CheckpointError("Incomplete checkpoint; run_id is unavailable")

    mismatches: list[dict[str, Any]] = []
    if expected_provenance is not None:
        actual = checkpoint.get("provenance") or {}
        for field, expected in expected_provenance.items():
            expected_hash = expected.get("sha256") if isinstance(expected, Mapping) else None
            observed = actual.get(field)
            observed_hash = observed.get("sha256") if isinstance(observed, Mapping) else None
            if expected_hash != observed_hash:
                resume_compatible = False
                if field == "resolved_config" and isinstance(expected, Mapping):
                    expected_resume_hash = expected.get("resume_sha256")
                    observed_resume_hash = (
                        observed.get("resume_sha256")
                        if isinstance(observed, Mapping)
                        else None
                    )
                    if observed_resume_hash is None and isinstance(
                        checkpoint.get("resolved_config"), Mapping
                    ):
                        observed_resume_hash = resolved_config_resume_hash(
                            checkpoint["resolved_config"]
                        )
                    resume_compatible = (
                        expected_resume_hash is not None
                        and expected_resume_hash == observed_resume_hash
                    )
                if resume_compatible:
                    continue
                mismatches.append(
                    {"field": field, "checkpoint": observed_hash, "current": expected_hash}
                )
    if mismatches and not allow_mismatch:
        details = ", ".join(item["field"] for item in mismatches)
        raise CheckpointMismatchError(
            f"Checkpoint provenance mismatch in: {details}. "
            "Pass the explicit mismatch override only after verifying the inputs."
        )
    return mismatches


def write_checkpoint_pointer(run_dir: str | Path, target: str | Path) -> Path:
    run_dir = Path(run_dir)
    target = Path(target)
    try:
        relative = target.relative_to(run_dir)
    except ValueError as exc:
        raise CheckpointError("Checkpoint pointer target must be inside the run directory") from exc
    pointer = run_dir / POINTER_NAME
    atomic_json_save(
        {
            "pointer_schema": {"name": "cardiodit.checkpoint-pointer", "version": 1},
            "target": str(relative),
        },
        pointer,
    )
    return pointer


def _periodic_candidates(run_dir: Path) -> list[Path]:
    candidates = []
    for path in [
        *run_dir.glob("checkpoint_epoch_*.pth"),
        *run_dir.glob("checkpoint_update_*.pth"),
    ]:
        try:
            number = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((number, path))
    return [path for _, path in sorted(candidates, reverse=True)]


def _pointer_target(run_dir: Path) -> Path | None:
    pointer = run_dir / POINTER_NAME
    if not pointer.is_file():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    target = (run_dir / payload["target"]).resolve()
    try:
        target.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise CheckpointError("Checkpoint pointer escapes the run directory") from exc
    return target


def load_latest_checkpoint(
    run_dir: str | Path,
    expected_provenance: Mapping[str, Any] | None = None,
    *,
    allow_mismatch: bool = False,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]] | None:
    """Try the pointer then newest numerical epoch/update checkpoint, skipping corruption."""
    run_dir = Path(run_dir)
    candidates: list[Path] = []
    try:
        pointer_target = _pointer_target(run_dir)
    except (OSError, ValueError, KeyError, CheckpointError):
        pointer_target = None
    if pointer_target is not None:
        candidates.append(pointer_target)

    # Read old full last_checkpoint.pth only when no modern pointer exists.
    legacy_last = run_dir / "last_checkpoint.pth"
    if pointer_target is None and legacy_last.is_file():
        candidates.append(legacy_last)
    candidates.extend(path for path in _periodic_candidates(run_dir) if path not in candidates)

    mismatch_error: CheckpointMismatchError | None = None
    for candidate in candidates:
        try:
            raw = torch.load(candidate, map_location="cpu", weights_only=False)
            checkpoint = adapt_legacy_checkpoint(raw)
            mismatches = validate_checkpoint(
                checkpoint, expected_provenance, allow_mismatch=allow_mismatch
            )
            return checkpoint, candidate, mismatches
        except CheckpointMismatchError as exc:
            mismatch_error = exc
            # A provenance mismatch is semantic, not corruption.  Periodic files
            # from the same run will not fix it, so preserve the actionable error.
            break
        except (
            OSError,
            EOFError,
            ValueError,
            RuntimeError,
            KeyError,
            pickle.UnpicklingError,
            CheckpointError,
        ):
            continue
    if mismatch_error is not None:
        raise mismatch_error
    return None

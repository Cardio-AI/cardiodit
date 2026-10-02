"""Strict, centralized Stage-1 model loading and identity construction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from src.models.vqvae import VQVAE
from src.utils.checkpointing import canonical_hash, sha256_file


class Stage1LoadError(RuntimeError):
    """Raised when a Stage-1 artifact is incomplete or ambiguous."""


def resolved_stage1_config(config_path: str | Path):
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 config does not exist: {path}")
    config = OmegaConf.load(path)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping) or not isinstance(resolved.get("model"), Mapping):
        raise Stage1LoadError(f"Stage-1 config has no model section: {path}")
    if not isinstance(resolved["model"].get("params"), Mapping):
        raise Stage1LoadError(f"Stage-1 config has no model.params section: {path}")
    return config, dict(resolved), path


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Full training checkpoints contain trusted local RNG/config objects
        # that the weights-only unpickler intentionally rejects.
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_model_state(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise Stage1LoadError("Stage-1 checkpoint payload must be a mapping")
    for key in ("model", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            return state
    if checkpoint and all(
        isinstance(key, str) and isinstance(value, (torch.Tensor, torch.nn.Parameter))
        for key, value in checkpoint.items()
    ):
        return checkpoint
    raise Stage1LoadError(
        "Stage-1 checkpoint has no model/state_dict and is not a raw state dict"
    )


def _normalize_ddp_prefix(state: Mapping[str, Any], expected_keys: set[str]):
    keys = set(state)
    if keys == expected_keys:
        return dict(state)
    if keys and all(key.startswith("module.") for key in keys):
        stripped = {key[7:]: value for key, value in state.items()}
        if set(stripped) == expected_keys:
            return stripped
    return dict(state)


def load_stage1_strict(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
):
    """Construct Stage-1 from its resolved config and require 100% key coverage.

    Returns ``(model, config, identity)``. The identity contains canonical
    resolved-config and byte-level artifact hashes for latent provenance.
    """
    config, resolved, config_path = resolved_stage1_config(config_path)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint_path}")

    model = VQVAE(**config.model.params)
    checkpoint = _load_checkpoint(checkpoint_path)
    state = _normalize_ddp_prefix(
        _extract_model_state(checkpoint), set(model.state_dict())
    )
    expected = set(model.state_dict())
    observed = set(state)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise Stage1LoadError(
            "Stage-1 checkpoint does not provide 100% expected key coverage: "
            f"missing={missing[:12]}{'...' if len(missing) > 12 else ''}, "
            f"unexpected={unexpected[:12]}{'...' if len(unexpected) > 12 else ''}"
        )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise Stage1LoadError(f"Stage-1 state tensor mismatch: {error}") from error

    identity = {
        "config_sha256": canonical_hash(resolved),
        "config_file_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_name": config_path.name,
        "checkpoint_name": checkpoint_path.name,
    }
    model = model.to(device).eval().requires_grad_(False)
    return model, config, identity

"""Utilities for loading 1D cardiac motion descriptor sidecars."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


DESCRIPTOR_KEYS = ("descriptor", "alpha_t", "descriptor_template", "template")
TEMPLATE_KEYS = ("descriptor_template", "template", "descriptor", "alpha_t")


def descriptor_sidecar_candidates(sidecar_dir: str | Path, subject_id: str) -> list[Path]:
    """Return supported descriptor sidecar paths in lookup priority order."""
    root = Path(sidecar_dir)
    stems = [
        root / "alpha_t" / subject_id,
        root / "descriptors" / subject_id,
        root / "motion_descriptor" / subject_id,
        root / "motion_descriptors" / subject_id,
        root / subject_id,
        root / f"{subject_id}.alpha",
    ]
    return [Path(f"{stem}{suffix}") for stem in stems for suffix in (".pt", ".npy")]


def extract_1d_tensor(
    payload,
    path: str | Path,
    kind: str,
    *,
    keys: Iterable[str] = DESCRIPTOR_KEYS,
) -> torch.Tensor:
    """Extract and validate a finite 1D tensor from a tensor, ndarray, or dict."""
    path = Path(path)
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, np.ndarray):
        tensor = torch.as_tensor(payload)
    elif isinstance(payload, dict):
        tensor = None
        for key in keys:
            if key in payload:
                tensor = payload[key]
                break
        if tensor is None:
            joined = ", ".join(keys)
            raise ValueError(f"{kind} sidecar {path} must contain one of: {joined}")
        if isinstance(tensor, np.ndarray):
            tensor = torch.as_tensor(tensor)
    else:
        raise ValueError(
            f"{kind} sidecar {path} must be a tensor, ndarray, or tensor-containing dict, "
            f"got {type(payload).__name__}"
        )

    if not torch.is_tensor(tensor):
        raise ValueError(
            f"{kind} sidecar {path} resolved to {type(tensor).__name__}, not a tensor"
        )
    if tensor.ndim != 1:
        raise ValueError(
            f"{kind} sidecar {path} must be 1D, got shape {tuple(tensor.shape)}"
        )
    tensor = tensor.detach().cpu().float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{kind} sidecar {path} contains non-finite values")
    return tensor


def load_1d_tensor(
    path: str | Path,
    kind: str = "descriptor",
    *,
    keys: Iterable[str] = DESCRIPTOR_KEYS,
) -> torch.Tensor:
    """Load a finite 1D tensor from a supported descriptor file.

    Supported files:
    - ``.pt`` tensor or tensor-containing dict
    - ``.npy`` numeric NumPy array
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload = np.load(path, allow_pickle=False)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return extract_1d_tensor(payload, path, kind, keys=keys)


def find_descriptor_sidecar(sidecar_dir: str | Path, subject_id: str) -> Path:
    candidates = descriptor_sidecar_candidates(sidecar_dir, subject_id)
    for path in candidates:
        if path.exists():
            return path
    tried = " and ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing descriptor for '{subject_id}': tried {tried}"
    )


def load_descriptor_sidecar(
    sidecar_dir: str | Path,
    subject_id: str,
    kind: str = "descriptor",
) -> tuple[torch.Tensor, Path]:
    path = find_descriptor_sidecar(sidecar_dir, subject_id)
    return load_1d_tensor(path, kind, keys=DESCRIPTOR_KEYS), path

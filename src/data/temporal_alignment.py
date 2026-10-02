from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DESCRIPTOR_KEYS = ("alpha_t", "phase", "cardiac_phase", "descriptor")
SIDECAR_SUFFIXES = (".pt", ".pth", ".npz", ".npy", ".json", ".csv")


def _as_1d_float_tensor(values) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        raise ValueError("Temporal descriptor is empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("Temporal descriptor contains non-finite values")
    return tensor


def normalize_phase(alpha_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map an arbitrary monotone descriptor to [0, 1]."""
    alpha_t = _as_1d_float_tensor(alpha_t)
    lo = alpha_t.min()
    hi = alpha_t.max()
    if (hi - lo).abs() < eps:
        return torch.zeros_like(alpha_t)
    return (alpha_t - lo) / (hi - lo)


def linear_phase(length: int, endpoint: bool = True) -> torch.Tensor:
    """Default descriptor for a uniformly sampled cardiac cycle."""
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return torch.zeros(1, dtype=torch.float32)
    end = 1.0 if endpoint else 1.0 - 1.0 / length
    return torch.linspace(0.0, end, length, dtype=torch.float32)


def _load_csv_descriptor(path: Path) -> torch.Tensor:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            key = next((k for k in DESCRIPTOR_KEYS if k in reader.fieldnames), None)
            field_is_numeric = False
            if len(reader.fieldnames) == 1:
                try:
                    float(reader.fieldnames[0])
                    field_is_numeric = True
                except ValueError:
                    field_is_numeric = False
            if key is None and len(reader.fieldnames) == 1 and not field_is_numeric:
                key = reader.fieldnames[0]
            if key is not None:
                return _as_1d_float_tensor([float(row[key]) for row in reader])

    rows = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return _as_1d_float_tensor(rows)


def _extract_descriptor(obj, path: Path) -> torch.Tensor:
    if isinstance(obj, Mapping):
        for key in DESCRIPTOR_KEYS:
            if key in obj:
                return _as_1d_float_tensor(obj[key])
        raise KeyError(
            f"No descriptor key found in {path}. Expected one of {DESCRIPTOR_KEYS}."
        )
    return _as_1d_float_tensor(obj)


def load_descriptor_sidecar(path: str | Path) -> torch.Tensor:
    """
    Load a 1D temporal descriptor sidecar.

    Supported formats:
    - ``.pt`` / ``.pth``: tensor, array, list, or dict with alpha/phase key
    - ``.npy``: 1D array
    - ``.npz``: alpha/phase key or the sole stored array
    - ``.json``: list or dict with alpha/phase key
    - ``.csv``: alpha/phase column, one-column CSV, or numeric CSV
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        return _extract_descriptor(torch.load(path, map_location="cpu", weights_only=False), path)
    if suffix == ".npy":
        return _as_1d_float_tensor(np.load(path))
    if suffix == ".npz":
        data = np.load(path)
        key = next((k for k in DESCRIPTOR_KEYS if k in data.files), None)
        if key is None:
            if len(data.files) != 1:
                raise KeyError(
                    f"No descriptor key found in {path}. Expected one of {DESCRIPTOR_KEYS}."
                )
            key = data.files[0]
        return _as_1d_float_tensor(data[key])
    if suffix == ".json":
        with path.open() as f:
            return _extract_descriptor(json.load(f), path)
    if suffix == ".csv":
        return _load_csv_descriptor(path)
    raise ValueError(f"Unsupported temporal descriptor format: {path.suffix}")


def resolve_descriptor_path(
    image_path: str | Path,
    phase_dir: str | Path | None,
    suffixes: Sequence[str] = SIDECAR_SUFFIXES,
) -> Path | None:
    """Find a descriptor sidecar by latent/image stem inside ``phase_dir``."""
    if phase_dir is None:
        return None
    phase_dir = Path(phase_dir)
    stem = Path(image_path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    if stem.endswith(".nii"):
        stem = stem[:-4]

    for suffix in suffixes:
        candidate = phase_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def pool_alpha_t(
    alpha_t: torch.Tensor | Sequence[float],
    target_length: int,
    method: str | None = "linear",
) -> torch.Tensor:
    """
    Resize a descriptor to latent temporal length.

    ``linear`` uses 1D interpolation and is the default for descriptors sampled
    at image frame rate. ``mean`` averages contiguous bins. ``none`` requires
    the input length to already match ``target_length``.
    """
    alpha = _as_1d_float_tensor(alpha_t)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if alpha.numel() == target_length:
        return alpha

    method = "linear" if method is None else str(method).lower()
    if method in ("none", "identity"):
        raise ValueError(
            f"alpha_t length {alpha.numel()} does not match target_length {target_length}"
        )
    if method in ("linear", "interp", "interpolate"):
        x = alpha.view(1, 1, -1)
        out = F.interpolate(x, size=target_length, mode="linear", align_corners=True)
        return out.view(-1)
    if method in ("mean", "avg", "average"):
        edges = torch.linspace(0, alpha.numel(), target_length + 1)
        pooled = []
        for start, end in zip(edges[:-1], edges[1:]):
            lo = int(torch.floor(start).item())
            hi = int(torch.ceil(end).item())
            pooled.append(alpha[lo:max(hi, lo + 1)].mean())
        return torch.stack(pooled)
    raise ValueError(f"Unknown alpha pooling method '{method}'")


def pairwise_l1_cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()
    return (a[:, None] - b[None, :]).abs()


def dtw_path(
    x: torch.Tensor | Sequence[float],
    y: torch.Tensor | Sequence[float],
    window: int | None = None,
) -> list[tuple[int, int]]:
    """Classic dynamic-time-warping path for two 1D descriptors."""
    cost = pairwise_l1_cost(_as_1d_float_tensor(x), _as_1d_float_tensor(y))
    n, m = cost.shape
    acc = torch.full((n + 1, m + 1), float("inf"), dtype=torch.float32)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        j_min = 1 if window is None else max(1, i - window)
        j_max = m if window is None else min(m, i + window)
        for j in range(j_min, j_max + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j],
                acc[i, j - 1],
                acc[i - 1, j - 1],
            )

    if not torch.isfinite(acc[n, m]):
        raise ValueError("No DTW path found; increase the Sakoe-Chiba window")

    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 or j > 0:
        path.append((max(i - 1, 0), max(j - 1, 0)))
        choices = (
            acc[i - 1, j - 1] if i > 0 and j > 0 else torch.tensor(float("inf")),
            acc[i - 1, j] if i > 0 else torch.tensor(float("inf")),
            acc[i, j - 1] if j > 0 else torch.tensor(float("inf")),
        )
        step = int(torch.argmin(torch.stack([c.float() for c in choices])))
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return path


def warp_sequence_to_template(
    sequence: torch.Tensor | Sequence[float],
    descriptor: torch.Tensor | Sequence[float],
    template: torch.Tensor | Sequence[float],
    window: int | None = None,
) -> torch.Tensor:
    """Warp a 1D sequence onto a template descriptor using DTW averaging."""
    seq = _as_1d_float_tensor(sequence)
    desc = _as_1d_float_tensor(descriptor)
    tmpl = _as_1d_float_tensor(template)
    if seq.numel() != desc.numel():
        raise ValueError("sequence and descriptor must have the same length")

    path = dtw_path(desc, tmpl, window=window)
    buckets: list[list[torch.Tensor]] = [[] for _ in range(tmpl.numel())]
    for src_idx, dst_idx in path:
        buckets[dst_idx].append(seq[src_idx])

    warped = []
    for idx, bucket in enumerate(buckets):
        if bucket:
            warped.append(torch.stack(bucket).mean())
        else:
            warped.append(seq[min(idx, seq.numel() - 1)])
    return torch.stack(warped)


def build_dtw_template(
    descriptors: Iterable[torch.Tensor | Sequence[float]],
    target_length: int | None = None,
    n_iters: int = 3,
    window: int | None = None,
) -> torch.Tensor:
    """
    Build a simple barycenter template from descriptor curves.

    This is intentionally lightweight: initialize with the median descriptor
    length or ``target_length``, align every descriptor to the current template,
    then average warped descriptors for ``n_iters`` rounds.
    """
    descs = [normalize_phase(_as_1d_float_tensor(d)) for d in descriptors]
    if not descs:
        raise ValueError("No descriptors provided")
    if target_length is None:
        lengths = sorted(d.numel() for d in descs)
        target_length = lengths[len(lengths) // 2]

    template = linear_phase(target_length)
    for _ in range(max(1, n_iters)):
        warped = [
            warp_sequence_to_template(d, d, template, window=window)
            for d in descs
        ]
        template = normalize_phase(torch.stack(warped).mean(dim=0))
    return template


def align_by_descriptor(
    sequence: torch.Tensor | Sequence[float],
    descriptor: torch.Tensor | Sequence[float],
    target_length: int,
    method: str = "linear",
) -> torch.Tensor:
    """Convenience wrapper to resample a 1D sequence using descriptor order."""
    seq = _as_1d_float_tensor(sequence)
    desc = normalize_phase(descriptor)
    if seq.numel() != desc.numel():
        raise ValueError("sequence and descriptor must have the same length")
    if method == "linear":
        order = torch.argsort(desc)
        sorted_seq = seq[order]
        return pool_alpha_t(sorted_seq, target_length, method="linear")
    if method == "dtw":
        template = linear_phase(target_length)
        return warp_sequence_to_template(seq, desc, template)
    if method == "motionfield":
        raise NotImplementedError(
            "motionfield temporal alignment is blocked until the phi_t sidecar "
            "format is finalized."
        )
    raise ValueError(f"Unknown temporal alignment method '{method}'")

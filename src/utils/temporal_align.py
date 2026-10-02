"""
Image-space temporal alignment for CardioDiT-TempAlign.

All functions operate on tensors whose last dimension is the temporal axis and
return tensors of the same rank with the last dimension equal to ``T_out``.

Methods dispatched by :func:`align`:

================  ============================================================
``cyclic``        Repeat-then-crop (current CardioDiT baseline; Exp 1).
``linear``        Linear interpolation along T (Exp 2).
``fourier``       rFFT zero-pad / truncate resample (Exp 4).
``piecewise``     Piecewise-linear keyframe alignment (Exp 3, sidecar).
``dtw``           Descriptor-DTW to a canonical template (Exps 5, 6, sidecar).
``motionfield``   Dense displacement-field warp (Exp 7, sidecar).
``native``        Pad T up to the nearest multiple of ``t_multiple``
                  (Exps 8, 9, 10). ``t_multiple`` must be ``vqgan_temporal_stride
                  * patch_size_t`` so the resulting latent T stays divisible
                  by the DiT temporal patch size; otherwise PatchEmbed4D
                  rejects the batch.  For the canonical ``best.yaml``
                  (stride 4, ``pt=2``) the correct value is 8.
================  ============================================================

The piecewise, dtw, and motionfield branches require sidecar data from the
Mueller et al. cmr-multi-view-phase-detection pipeline. Piecewise keyframes and
descriptor-DTW are implemented; motionfield remains blocked until φ_t sidecars
are finalised.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import torch
import torch.nn.functional as F

PIECEWISE_PHASE_ORDER = ("ED", "MS", "ES", "PF", "MD")

# Derived from the provided training keyframes on 2026-06-09.  Mean normalised
# train-cycle positions round to output frames 0/6/12/19/28 for T_out=32.
DEFAULT_PIECEWISE_TARGETS = {
    "ED": 0.0,
    "MS": 6.0 / 32.0,
    "ES": 12.0 / 32.0,
    "PF": 19.0 / 32.0,
    "MD": 28.0 / 32.0,
}


def cyclic_repeat(x: torch.Tensor, T_out: int) -> torch.Tensor:
    T_in = x.shape[-1]
    if T_in == T_out:
        return x
    if T_in < T_out:
        n_repeats = (T_out + T_in - 1) // T_in
        repeats = [1] * x.ndim
        repeats[-1] = n_repeats
        x = x.repeat(*repeats)
    return x[..., :T_out]


def linear_interp(x: torch.Tensor, T_out: int) -> torch.Tensor:
    T_in = x.shape[-1]
    if T_in == T_out:
        return x
    lead_shape = x.shape[:-1]
    flat = x.reshape(1, -1, T_in)
    out = F.interpolate(flat, size=T_out, mode="linear", align_corners=False)
    return out.reshape(*lead_shape, T_out)


def fourier_resample(x: torch.Tensor, T_out: int) -> torch.Tensor:
    T_in = x.shape[-1]
    if T_in == T_out:
        return x
    F_in = torch.fft.rfft(x, dim=-1)
    n_in = T_in // 2 + 1
    n_out = T_out // 2 + 1
    if n_out <= n_in:
        F_out = F_in[..., :n_out]
    else:
        pad = [0] * (2 * x.ndim)
        pad[1] = n_out - n_in
        F_out = F.pad(F_in, pad)
    # torch.fft.irfft uses "backward" norm (divides by n), and rfft uses no
    # norm, so a same-length round-trip is identity. When resampling we must
    # rescale by T_out / T_in to keep amplitudes consistent — mirrors
    # ``scipy.signal.resample``'s convention.
    out = torch.fft.irfft(F_out, n=T_out, dim=-1)
    return out * (T_out / T_in)


def pad_to_multiple_of(x: torch.Tensor, m: int) -> torch.Tensor:
    if m < 1:
        raise ValueError(f"pad_to_multiple_of: m must be >= 1, got {m}")
    T_in = x.shape[-1]
    T_pad = ((T_in + m - 1) // m) * m
    if T_pad == T_in:
        return x
    return cyclic_repeat(x, T_pad)


def piecewise_interp(
    x: torch.Tensor,
    T_out: int,
    keyframes: Mapping[str, int],
    targets: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    """
    Linearly interpolate within each adjacent keyframe pair such that the five
    detected cardiac keyframes land on fixed normalised positions in the output.

    ``keyframes`` maps phase name -> source frame index in ``x``. Expected keys:
    ``ED, MS, ES, PF, MD``.  The cardiac cycle is treated circularly: ED anchors
    output frame 0, and phases after ED may wrap around the native frame axis.

    ``targets`` maps the same names to normalised cycle positions in ``[0, 1)``
    or absolute output positions in ``[0, T_out)``. When omitted, training-set
    mean positions from ``data/keyframes`` are used.
    """
    if T_out < 2:
        raise ValueError(f"piecewise_interp: T_out must be >= 2, got {T_out}")
    T_in = x.shape[-1]
    if T_in < 2:
        raise ValueError(f"piecewise_interp: input T must be >= 2, got {T_in}")

    source_anchors = _piecewise_source_anchors(keyframes, T_in)
    target_anchors = _piecewise_target_anchors(T_out, targets)

    out_pos = torch.arange(T_out, device=x.device, dtype=torch.float32)
    target = torch.tensor(target_anchors, device=x.device, dtype=torch.float32)
    source = torch.tensor(source_anchors, device=x.device, dtype=torch.float32)

    # Segment i maps [target_i, target_{i+1}] to [source_i, source_{i+1}].
    seg = torch.bucketize(out_pos, target[1:], right=True)
    left_t = target[seg]
    right_t = target[seg + 1]
    left_s = source[seg]
    right_s = source[seg + 1]
    frac = (out_pos - left_t) / (right_t - left_t)
    source_pos = left_s + frac * (right_s - left_s)

    return _sample_periodic_linear(x, source_pos)


def dtw_align(
    x: torch.Tensor,
    T_out: int,
    descriptor: torch.Tensor,
    template: torch.Tensor,
    *,
    keyframes: Optional[Mapping[str, int]] = None,
    template_keyframe_positions: Optional[Mapping[str, float]] = None,
    cyclic: bool = True,
    max_warp_fraction: float = 0.25,
    non_diagonal_penalty: float = 1e-3,
) -> torch.Tensor:
    """
    Align a cine sequence to a canonical descriptor template using constrained
    dynamic time warping, then sample the original image sequence at the
    resulting source-time grid.

    ``descriptor`` is the per-subject 1D cardiac motion trace with length equal
    to the input image T. ``template`` is the canonical 1D trace with length
    ``T_out``. DTW is performed only on these compact descriptors; raw image
    tensors are never used for the DTW cost. The final source grid is applied to
    ``x`` with the same periodic linear sampler used by piecewise alignment.

    If ``keyframes`` and ``template_keyframe_positions`` are supplied, DTW runs
    independently inside ED→MS→ES→PF→MD→next-ED intervals. This is the preferred
    mode for templates built by :func:`build_dtw_template` because it prevents
    systolic and diastolic segments from matching each other.

    Without keyframe anchors, cardiac cine is still treated as cyclic. By
    default, all integer rotations of the subject descriptor are evaluated and
    the lowest-cost constrained path is selected. The winning rotation defines
    the explicit cycle boundary: output/template frame 0 samples from that
    source position modulo ``T_in``. Set ``cyclic=False`` only when descriptors
    are already ED/cycle-boundary aligned.

    Pathologies are limited by two simple constraints suitable for small T:
    a Sakoe-Chiba-style band in normalised phase coordinates and a small penalty
    on horizontal/vertical DTW moves, which discourages repeated/skipped runs
    when descriptor values are tied or nearly flat.
    """
    T_in = x.shape[-1]
    if T_out < 2:
        raise ValueError(f"dtw_align: T_out must be >= 2, got {T_out}")
    if T_in < 2:
        raise ValueError(f"dtw_align: input T must be >= 2, got {T_in}")

    desc = _validate_dtw_vector(descriptor, "descriptor")
    tmpl = _validate_dtw_vector(template, "template")
    if desc.numel() != T_in:
        raise ValueError(
            f"dtw_align: descriptor length {desc.numel()} does not match input T={T_in}"
        )
    if tmpl.numel() != T_out:
        raise ValueError(
            f"dtw_align: template length {tmpl.numel()} does not match T_out={T_out}"
        )
    if not (0.0 <= float(max_warp_fraction) <= 1.0):
        raise ValueError(
            "dtw_align: max_warp_fraction must be in [0, 1], "
            f"got {max_warp_fraction}"
        )
    if float(non_diagonal_penalty) < 0.0:
        raise ValueError(
            "dtw_align: non_diagonal_penalty must be >= 0, "
            f"got {non_diagonal_penalty}"
        )

    desc = _normalise_dtw_vector(desc)
    tmpl = _normalise_dtw_vector(tmpl)

    if keyframes is not None or template_keyframe_positions is not None:
        if keyframes is None or template_keyframe_positions is None:
            raise ValueError(
                "dtw_align: keyframes and template_keyframe_positions must be "
                "provided together"
            )
        source_pos = _anchored_dtw_source_positions(
            desc,
            tmpl,
            keyframes,
            template_keyframe_positions,
            max_warp_fraction=float(max_warp_fraction),
            non_diagonal_penalty=float(non_diagonal_penalty),
        )
        return _sample_periodic_linear(x, source_pos.to(device=x.device))

    shifts = range(T_in) if cyclic else range(1)
    best_cost = math.inf
    best_shift = 0
    best_path: list[tuple[int, int]] | None = None
    for shift in shifts:
        rolled = torch.roll(desc, shifts=-shift, dims=0)
        cost, path = _dtw_path(
            rolled,
            tmpl,
            max_warp_fraction=float(max_warp_fraction),
            non_diagonal_penalty=float(non_diagonal_penalty),
        )
        if cost < best_cost:
            best_cost = cost
            best_shift = shift
            best_path = path

    if best_path is None:
        raise ValueError("dtw_align: failed to compute a valid DTW path")

    source_pos = _dtw_path_to_source_positions(best_path, T_in, T_out)
    source_pos = source_pos + float(best_shift)
    return _sample_periodic_linear(x, source_pos.to(device=x.device))


def build_dtw_template(
    descriptors: list[torch.Tensor],
    keyframes: list[Mapping[str, int]],
    *,
    T_out: int = 32,
    n_iters: int = 5,
    aggregation: str = "median",
    max_warp_fraction: float = 0.5,
    non_diagonal_penalty: float = 1e-3,
) -> dict:
    """
    Build the Exp 5 canonical temporal template from training descriptors.

    The template is intentionally temporal-only: it stores the learned
    32-frame descriptor trajectory plus the population-average keyframe
    positions. Images, masks, and deformation fields should be resampled to
    this temporal template later; they are not part of the core template.

    Construction:
    1. Validate each 1D subject descriptor and keyframe set.
    2. Compute per-subject ED→MS→ES→PF→MD→next-ED durations and take the
       robust population median.
    3. Place continuous keyframe positions on a 32-frame cyclic template grid.
    4. Piecewise-warp each z-normalised descriptor to that grid and aggregate.
    5. Optionally refine the descriptor template with interval-wise constrained
       DTW inside each keyframe interval.
    """
    if len(descriptors) != len(keyframes):
        raise ValueError(
            "build_dtw_template: descriptors and keyframes must have the same "
            f"length, got {len(descriptors)} and {len(keyframes)}"
        )
    if not descriptors:
        raise ValueError("build_dtw_template: at least one descriptor is required")
    if T_out < len(PIECEWISE_PHASE_ORDER) + 1:
        raise ValueError(
            "build_dtw_template: T_out is too small for five phase intervals, "
            f"got {T_out}"
        )
    if n_iters < 0:
        raise ValueError(f"build_dtw_template: n_iters must be >= 0, got {n_iters}")
    if aggregation not in {"median", "mean"}:
        raise ValueError(
            "build_dtw_template: aggregation must be 'median' or 'mean', "
            f"got {aggregation}"
        )
    if not (0.0 <= float(max_warp_fraction) <= 1.0):
        raise ValueError(
            "build_dtw_template: max_warp_fraction must be in [0, 1], "
            f"got {max_warp_fraction}"
        )
    if float(non_diagonal_penalty) < 0.0:
        raise ValueError(
            "build_dtw_template: non_diagonal_penalty must be >= 0, "
            f"got {non_diagonal_penalty}"
        )

    norm_descs = []
    duration_rows = []
    for idx, (descriptor, kf) in enumerate(zip(descriptors, keyframes)):
        desc = _validate_dtw_vector(descriptor, f"descriptor[{idx}]")
        anchors = _piecewise_source_anchors(kf, desc.numel())
        durations = torch.tensor(
            [right - left for left, right in zip(anchors[:-1], anchors[1:])],
            dtype=torch.float32,
        )
        duration_rows.append(durations / float(desc.numel()))
        norm_descs.append(_normalise_dtw_vector(desc))

    ratios = _aggregate_stack(torch.stack(duration_rows), aggregation)
    ratios = ratios / ratios.sum()
    keyframe_positions = _template_keyframe_positions(ratios, T_out)
    piecewise_targets = {
        phase: keyframe_positions[phase]
        for phase in PIECEWISE_PHASE_ORDER
    }

    warped = [
        piecewise_interp(desc.view(1, -1), T_out, keyframes=kf, targets=piecewise_targets)
        .view(T_out)
        for desc, kf in zip(norm_descs, keyframes)
    ]
    template = _aggregate_stack(torch.stack(warped), aggregation)

    template_bounds = _integer_template_bounds(keyframe_positions, T_out)
    for _ in range(n_iters):
        template = _refine_dtw_template_once(
            norm_descs,
            keyframes,
            template,
            template_bounds,
            aggregation=aggregation,
            max_warp_fraction=max_warp_fraction,
            non_diagonal_penalty=non_diagonal_penalty,
        )

    return {
        "n_frames": int(T_out),
        "phase_grid": torch.arange(T_out, dtype=torch.float32) / float(T_out),
        "keyframe_positions": keyframe_positions,
        "keyframe_order": list(PIECEWISE_PHASE_ORDER) + ["ED_next"],
        "descriptor_template": template.detach().cpu().float(),
        "normalization": {
            "descriptor": "per_subject_zscore",
            "aggregation": aggregation,
        },
        "dtw_constraints": {
            "mode": "interval",
            "max_warp_fraction": float(max_warp_fraction),
            "non_diagonal_penalty": float(non_diagonal_penalty),
            "n_iters": int(n_iters),
            "template_integer_bounds": template_bounds,
        },
    }


def _piecewise_source_anchors(keyframes: Mapping[str, int], T_in: int) -> list[float]:
    missing = [p for p in PIECEWISE_PHASE_ORDER if p not in keyframes]
    if missing:
        raise ValueError(f"piecewise_interp: missing keyframes {missing}")

    raw = []
    for phase in PIECEWISE_PHASE_ORDER:
        frame = int(keyframes[phase])
        if frame < 0 or frame >= T_in:
            raise ValueError(
                f"piecewise_interp: keyframe {phase}={frame} outside [0, {T_in})"
            )
        raw.append(float(frame))

    anchors = [raw[0]]
    for frame in raw[1:]:
        while frame <= anchors[-1]:
            frame += T_in
        anchors.append(frame)
    anchors.append(anchors[0] + T_in)
    return anchors


def _piecewise_target_anchors(
    T_out: int,
    targets: Optional[Mapping[str, float]],
) -> list[float]:
    targets = DEFAULT_PIECEWISE_TARGETS if targets is None else targets
    missing = [p for p in PIECEWISE_PHASE_ORDER if p not in targets]
    if missing:
        raise ValueError(f"piecewise_interp: missing target positions {missing}")

    values = [float(targets[p]) for p in PIECEWISE_PHASE_ORDER]
    if all(0.0 <= v < 1.0 for v in values):
        anchors = [v * T_out for v in values]
    else:
        anchors = values

    if abs(anchors[0]) > 1e-6:
        raise ValueError(f"piecewise_interp: ED target must be 0, got {anchors[0]}")
    if anchors[-1] >= T_out:
        raise ValueError(
            f"piecewise_interp: final target must be < T_out={T_out}, got {anchors[-1]}"
        )
    for left, right in zip(anchors, anchors[1:]):
        if not left < right:
            raise ValueError(
                "piecewise_interp: target positions must be strictly increasing "
                f"in {PIECEWISE_PHASE_ORDER}, got {anchors}"
            )
    anchors.append(float(T_out))
    return anchors


def _aggregate_stack(values: torch.Tensor, aggregation: str) -> torch.Tensor:
    if aggregation == "median":
        return values.median(dim=0).values
    if aggregation == "mean":
        return values.mean(dim=0)
    raise ValueError(f"Unknown aggregation '{aggregation}'")


def _template_keyframe_positions(ratios: torch.Tensor, T_out: int) -> dict[str, float]:
    cumulative = torch.cumsum(ratios, dim=0) * float(T_out)
    return {
        "ED": 0.0,
        "MS": float(cumulative[0]),
        "ES": float(cumulative[1]),
        "PF": float(cumulative[2]),
        "MD": float(cumulative[3]),
        "ED_next": float(T_out),
    }


def _integer_template_bounds(
    keyframe_positions: Mapping[str, float],
    T_out: int,
) -> list[int]:
    raw = [0] + [int(round(float(keyframe_positions[p]))) for p in PIECEWISE_PHASE_ORDER[1:]] + [T_out]
    bounds = [0] * len(raw)
    bounds[0] = 0
    bounds[-1] = T_out
    n_intervals = len(raw) - 1
    for i in range(1, len(raw) - 1):
        min_allowed = bounds[i - 1] + 1
        max_allowed = T_out - (n_intervals - i)
        bounds[i] = min(max(raw[i], min_allowed), max_allowed)
    return bounds


def _refine_dtw_template_once(
    descriptors: list[torch.Tensor],
    keyframes: list[Mapping[str, int]],
    template: torch.Tensor,
    template_bounds: list[int],
    *,
    aggregation: str,
    max_warp_fraction: float,
    non_diagonal_penalty: float,
) -> torch.Tensor:
    contributions: list[list[torch.Tensor]] = [[] for _ in range(template.numel())]
    for desc, kf in zip(descriptors, keyframes):
        source_anchors = _piecewise_source_anchors(kf, desc.numel())
        for interval_idx, (j0, j1) in enumerate(zip(template_bounds[:-1], template_bounds[1:])):
            if j1 <= j0:
                continue
            source_start = int(round(source_anchors[interval_idx]))
            source_end = int(round(source_anchors[interval_idx + 1]))
            if source_end <= source_start:
                continue

            source_indices = torch.arange(source_start, source_end, dtype=torch.long)
            source_segment = desc[source_indices.remainder(desc.numel())]
            template_segment = template[j0:j1]
            if source_segment.numel() == 0 or template_segment.numel() == 0:
                continue

            _, path = _dtw_path(
                source_segment,
                template_segment,
                max_warp_fraction=max_warp_fraction,
                non_diagonal_penalty=non_diagonal_penalty,
            )
            for local_source_idx, local_template_idx in path:
                global_template_idx = j0 + local_template_idx
                contributions[global_template_idx].append(source_segment[local_source_idx])

    updated = template.clone()
    for idx, vals in enumerate(contributions):
        if not vals:
            continue
        stacked = torch.stack(vals)
        if aggregation == "median":
            updated[idx] = stacked.median()
        elif aggregation == "mean":
            updated[idx] = stacked.mean()
        else:
            raise ValueError(f"Unknown aggregation '{aggregation}'")
    return updated


def _anchored_dtw_source_positions(
    descriptor: torch.Tensor,
    template: torch.Tensor,
    keyframes: Mapping[str, int],
    template_keyframe_positions: Mapping[str, float],
    *,
    max_warp_fraction: float,
    non_diagonal_penalty: float,
) -> torch.Tensor:
    T_in = descriptor.numel()
    T_out = template.numel()
    source_anchors = _piecewise_source_anchors(keyframes, T_in)
    template_bounds = _integer_template_bounds(template_keyframe_positions, T_out)
    sums = torch.zeros(T_out, dtype=torch.float32)
    counts = torch.zeros(T_out, dtype=torch.float32)

    for interval_idx, (j0, j1) in enumerate(zip(template_bounds[:-1], template_bounds[1:])):
        source_start = int(round(source_anchors[interval_idx]))
        source_end = int(round(source_anchors[interval_idx + 1]))
        if j1 <= j0 or source_end <= source_start:
            continue

        source_indices = torch.arange(source_start, source_end, dtype=torch.long)
        source_segment = descriptor[source_indices.remainder(T_in)]
        template_segment = template[j0:j1]
        if source_segment.numel() == 0 or template_segment.numel() == 0:
            continue

        _, path = _dtw_path(
            source_segment,
            template_segment,
            max_warp_fraction=max_warp_fraction,
            non_diagonal_penalty=non_diagonal_penalty,
        )
        for local_source_idx, local_template_idx in path:
            global_template_idx = j0 + local_template_idx
            sums[global_template_idx] += float(source_indices[local_source_idx])
            counts[global_template_idx] += 1.0

    missing = counts == 0
    if bool(missing.any()):
        # Defensive fallback for degenerate intervals: use the keyframe-linear
        # map for frames not visited by any interval path.
        fallback = _piecewise_source_grid_from_targets(
            keyframes,
            template_keyframe_positions,
            T_in,
            T_out,
        )
        counts[missing] = 1.0
        sums[missing] = fallback[missing]
    return sums / counts


def _piecewise_source_grid_from_targets(
    keyframes: Mapping[str, int],
    targets: Mapping[str, float],
    T_in: int,
    T_out: int,
) -> torch.Tensor:
    source_anchors = _piecewise_source_anchors(keyframes, T_in)
    target_anchors = _piecewise_target_anchors(T_out, targets)
    out_pos = torch.arange(T_out, dtype=torch.float32)
    target = torch.tensor(target_anchors, dtype=torch.float32)
    source = torch.tensor(source_anchors, dtype=torch.float32)
    seg = torch.bucketize(out_pos, target[1:], right=True)
    left_t = target[seg]
    right_t = target[seg + 1]
    left_s = source[seg]
    right_s = source[seg + 1]
    frac = (out_pos - left_t) / (right_t - left_t)
    return left_s + frac * (right_s - left_s)


def _validate_dtw_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim != 1:
        raise ValueError(f"dtw_align: {name} must be a 1D tensor, got shape {tuple(x.shape)}")
    if x.numel() < 2:
        raise ValueError(f"dtw_align: {name} length must be >= 2, got {x.numel()}")
    x = x.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError(f"dtw_align: {name} must contain only finite values")
    return x


def _normalise_dtw_vector(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    std = x.std(unbiased=False)
    if float(std) > 1e-6:
        x = x / std
    return x


def _dtw_allowed_mask(
    T_source: int,
    T_template: int,
    max_warp_fraction: float,
) -> torch.Tensor:
    source_phase = torch.linspace(0.0, 1.0, T_source)
    template_phase = torch.linspace(0.0, 1.0, T_template)
    mask = (source_phase[:, None] - template_phase[None, :]).abs() <= max_warp_fraction
    mask[0, 0] = True
    mask[-1, -1] = True
    return mask


def _dtw_path(
    source: torch.Tensor,
    template: torch.Tensor,
    *,
    max_warp_fraction: float,
    non_diagonal_penalty: float,
) -> tuple[float, list[tuple[int, int]]]:
    T_source = source.numel()
    T_template = template.numel()
    dist = (source[:, None] - template[None, :]).square()
    allowed = _dtw_allowed_mask(T_source, T_template, max_warp_fraction)
    inf = torch.tensor(float("inf"), dtype=dist.dtype)
    dist = torch.where(allowed, dist, inf)

    dp = torch.full((T_source, T_template), float("inf"), dtype=dist.dtype)
    parent = torch.full((T_source, T_template), -1, dtype=torch.int8)
    dp[0, 0] = dist[0, 0]

    for i in range(T_source):
        for j in range(T_template):
            if i == 0 and j == 0:
                continue
            if not torch.isfinite(dist[i, j]):
                continue

            best = inf
            best_parent = -1
            candidates: list[tuple[torch.Tensor, int]] = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], 0))
            if i > 0:
                candidates.append((dp[i - 1, j] + non_diagonal_penalty, 1))
            if j > 0:
                candidates.append((dp[i, j - 1] + non_diagonal_penalty, 2))

            for value, move in candidates:
                if float(value) < float(best):
                    best = value
                    best_parent = move
            if best_parent >= 0 and torch.isfinite(best):
                dp[i, j] = dist[i, j] + best
                parent[i, j] = best_parent

    if not torch.isfinite(dp[-1, -1]):
        raise ValueError(
            "dtw_align: no valid DTW path under max_warp_fraction="
            f"{max_warp_fraction}; increase the constraint width"
        )

    path = []
    i = T_source - 1
    j = T_template - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        move = int(parent[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            raise ValueError("dtw_align: invalid DTW backtrace state")
    path.reverse()
    return float(dp[-1, -1]) / len(path), path


def _dtw_path_to_source_positions(
    path: list[tuple[int, int]],
    T_source: int,
    T_template: int,
) -> torch.Tensor:
    sums = torch.zeros(T_template, dtype=torch.float32)
    counts = torch.zeros(T_template, dtype=torch.float32)
    for source_idx, template_idx in path:
        sums[template_idx] += float(source_idx)
        counts[template_idx] += 1.0

    valid = counts > 0
    if not bool(valid.all()):
        # Standard 3-neighbour DTW visits every template index, but keep a
        # linear fill for defensive use if the step pattern is changed later.
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            raise ValueError("dtw_align: DTW path produced no template positions")
        pos = torch.empty(T_template, dtype=torch.float32)
        pos[valid] = sums[valid] / counts[valid]
        all_idx = torch.arange(T_template, dtype=torch.float32)
        for left, right in zip(valid_idx[:-1], valid_idx[1:]):
            gap = torch.arange(int(left) + 1, int(right), dtype=torch.long)
            if gap.numel() == 0:
                continue
            frac = (all_idx[gap] - float(left)) / float(right - left)
            pos[gap] = pos[left] + frac * (pos[right] - pos[left])
        pos[: valid_idx[0]] = pos[valid_idx[0]]
        pos[valid_idx[-1] + 1 :] = pos[valid_idx[-1]]
        return pos.clamp(0.0, float(T_source - 1))

    return (sums / counts).clamp(0.0, float(T_source - 1))


def _sample_periodic_linear(x: torch.Tensor, source_pos: torch.Tensor) -> torch.Tensor:
    T_in = x.shape[-1]
    left = torch.floor(source_pos).to(torch.long)
    right = left + 1
    weight = (source_pos - left.to(source_pos.dtype)).reshape(
        *([1] * (x.ndim - 1)), source_pos.numel()
    )
    if x.is_floating_point():
        weight = weight.to(dtype=x.dtype)

    x_left = torch.index_select(x, dim=-1, index=left.remainder(T_in))
    x_right = torch.index_select(x, dim=-1, index=right.remainder(T_in))
    return x_left + (x_right - x_left) * weight


def motionfield_warp(
    x: torch.Tensor,
    phi: torch.Tensor,
    T_out: int,
) -> torch.Tensor:
    """
    Resample ``x`` at ``T_out`` time points by warping each source frame ``i``
    forward by ``alpha * phi_i`` for fractional offset ``alpha`` (Mueller et al.
    Exp 7). ``phi`` may be ``(T_in, H, W, D, 3)`` for 3D or ``(T_in, H, W, 2)``
    for 2D — implementation must stay dim-agnostic.

    Blocked: confirm Mueller phi_t sidecar tensor layout, units (voxel vs
    normalised), and on-disk dtype before implementing.
    """
    raise NotImplementedError(
        "motionfield_warp blocked on Mueller et al. phi_t sidecar format."
    )


_ALIGN_FNS = {
    "cyclic": cyclic_repeat,
    "linear": linear_interp,
    "fourier": fourier_resample,
}


def align(
    x: torch.Tensor,
    method: str,
    T_out: int,
    **kwargs,
) -> torch.Tensor:
    if method == "native":
        # Default 8 = VQ-GAN temporal stride (4) * DiT patch size T (2).
        # Caller must override when the DiT patch size T differs.
        return pad_to_multiple_of(x, int(kwargs.get("t_multiple", 8)))
    if method == "piecewise":
        return piecewise_interp(x, T_out, keyframes=kwargs["keyframes"],
                                targets=kwargs.get("targets"))
    if method == "dtw":
        if "descriptor" not in kwargs or "template" not in kwargs:
            raise ValueError(
                "align(method='dtw') requires descriptor=... and template=..."
            )
        return dtw_align(
            x,
            T_out,
            descriptor=kwargs["descriptor"],
            template=kwargs["template"],
            keyframes=kwargs.get("keyframes"),
            template_keyframe_positions=kwargs.get("template_keyframe_positions"),
            cyclic=bool(kwargs.get("cyclic", True)),
            max_warp_fraction=float(kwargs.get("max_warp_fraction", 0.25)),
            non_diagonal_penalty=float(kwargs.get("non_diagonal_penalty", 1e-3)),
        )
    if method == "motionfield":
        return motionfield_warp(x, phi=kwargs["phi"], T_out=T_out)
    if method in _ALIGN_FNS:
        return _ALIGN_FNS[method](x, T_out)
    raise ValueError(
        f"Unknown temporal alignment method '{method}'. "
        "Expected one of: cyclic, linear, fourier, piecewise, dtw, "
        "motionfield, native."
    )

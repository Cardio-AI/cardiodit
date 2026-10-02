"""Fail-fast validation for scientific and operational configuration safety."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    return getter(key, default) if getter is not None else getattr(container, key, default)


def resolve_objective_loss(training: Any) -> str:
    """Resolve evolution and temporal-alignment loss names without ambiguity."""
    aliases = {
        "mse": "mse", "mse_loss": "mse", "l2": "mse",
        "l1": "l1", "mae": "l1",
        "huber": "huber_legacy", "huber_legacy": "huber_legacy",
        "smooth_l1": "huber_legacy", "smooth_l1_loss": "huber_legacy",
    }
    resolved = []
    for key in ("objective_loss", "loss_type"):
        value = _get(training, key)
        if value is not None:
            name = str(value).lower()
            if name not in aliases:
                raise ValueError(f"Unknown training.{key} {value!r}; expected {sorted(aliases)}")
            resolved.append(aliases[name])
    if len(set(resolved)) > 1:
        raise ValueError("training.objective_loss and training.loss_type select different losses")
    beta = float(_get(training, "huber_beta", 1.0))
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("training.huber_beta must be finite and nonnegative")
    return resolved[0] if resolved else "huber_legacy"


def estimate_attention(config: Any) -> dict[str, int]:
    """Return token and dense-attention estimates without constructing a model."""
    params = _get(_get(config, "model"), "params", {})
    input_size = tuple(
        int(v)
        for v in (_get(params, "max_input_size", None) or _get(params, "input_size"))
    )
    patch_size = tuple(int(v) for v in _get(params, "patch_size"))
    if len(input_size) != 4 or len(patch_size) != 4:
        raise ValueError("model input_size and patch_size must both contain Z, X, Y, T")
    if any(dim <= 0 or patch <= 0 for dim, patch in zip(input_size, patch_size)):
        raise ValueError("model input_size and patch_size values must be positive")
    if any(dim % patch for dim, patch in zip(input_size, patch_size)):
        raise ValueError(
            f"model input_size {input_size} must be divisible by patch_size {patch_size}"
        )

    grid = tuple(dim // patch for dim, patch in zip(input_size, patch_size))
    tokens = math.prod(grid)
    heads = int(_get(params, "num_heads", 1))
    depth = int(_get(params, "depth", 1))
    hidden = int(_get(params, "hidden_size", heads * 64))
    fused_attention = bool(_get(params, "flash_attention", False))
    batch = int(_get(_get(config, "training", {}), "batch_size", 1))
    score_lower_bound = batch * heads * tokens * tokens * 4
    # Conservative, not exact: one retained score-sized activation plus two
    # score-sized backward workspaces per layer. Efficient attention kernels can
    # use less; eager/global attention can use more due to QKV and allocator cost.
    dense_worst_case_bytes = score_lower_bound * depth * 3
    # Fused SDPA/Flash kernels do not materialize N² scores. This linear
    # activation/backward allowance is conservative for attention-specific QKV,
    # outputs and workspaces, but intentionally excludes MLP/optimizer memory.
    fused_training_bytes = batch * tokens * hidden * depth * 4 * 16
    configured_backend_bytes = (
        fused_training_bytes if fused_attention else dense_worst_case_bytes
    )
    return {
        "tokens": tokens,
        "grid_z": grid[0],
        "grid_x": grid[1],
        "grid_y": grid[2],
        "grid_t": grid[3],
        "attention_score_lower_bound_bytes": score_lower_bound,
        "dense_attention_worst_case_bytes": dense_worst_case_bytes,
        "fused_attention_conservative_bytes": fused_training_bytes,
        "configured_attention_backend": "fused_sdpa" if fused_attention else "dense",
        "conservative_training_attention_bytes": configured_backend_bytes,
        # Compatibility alias; explicitly a lower bound, not the gate estimate.
        "attention_score_bytes": score_lower_bound,
    }


def validate_dit_config(
    config: Any,
    *,
    legacy_checkpoint: Mapping[str, Any] | None = None,
    allow_legacy_scaling: bool = False,
) -> dict[str, int]:
    """Validate a new-run DiT config and return provenance-ready estimates."""
    params = _get(_get(config, "model"), "params", {})
    training = _get(config, "training", {})
    optim = _get(config, "optim", {})

    target_updates = _get(training, "target_optimizer_updates", None)
    checkpoint_updates = _get(training, "checkpoint_optimizer_updates", None)
    if (target_updates is None) != (checkpoint_updates is None):
        raise ValueError(
            "training.target_optimizer_updates and "
            "training.checkpoint_optimizer_updates must be declared together."
        )
    if target_updates is not None:
        target_updates = int(target_updates)
        checkpoint_updates = int(checkpoint_updates)
        if target_updates <= 0 or checkpoint_updates <= 0:
            raise ValueError("Optimizer-update targets and intervals must be positive.")
        if checkpoint_updates > target_updates:
            raise ValueError(
                "training.checkpoint_optimizer_updates cannot exceed the target."
            )
        schedule_updates = _get(optim, "total_updates", None)
        if schedule_updates is None or int(schedule_updates) != target_updates:
            raise ValueError(
                "optim.total_updates must equal "
                "training.target_optimizer_updates for an exact run."
            )
    resolve_objective_loss(training)

    primary_metric = _get(training, "validation_primary_metric", None)
    self_conditioning = bool(_get(training, "self_conditioning", False))
    valid_primary = {"unconditioned", "self_conditioned"} if self_conditioning else {"unconditioned"}
    if primary_metric is not None and str(primary_metric).lower() not in valid_primary:
        raise ValueError(
            "training.validation_primary_metric must name an evaluated inference "
            f"path; valid choices are {sorted(valid_primary)}."
        )

    if bool(_get(params, "learn_sigma", False)):
        raise ValueError(
            "learn_sigma=true is unsupported because no learned-variance loss or "
            "sampler exists; set model.params.learn_sigma=false."
        )

    normalize = bool(_get(training, "normalize_latents", False))
    scale = float(_get(training, "scale_factor", 1.0))
    if normalize and not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-8):
        checkpoint_proves_legacy = (
            allow_legacy_scaling
            and legacy_checkpoint is not None
            and bool(legacy_checkpoint.get("normalize_latents"))
            and math.isclose(
                float(legacy_checkpoint.get("scale_factor", float("nan"))),
                scale,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and legacy_checkpoint.get("latent_mean") is not None
            and legacy_checkpoint.get("latent_std") is not None
        )
        if not checkpoint_proves_legacy:
            raise ValueError(
                "Ambiguous latent scaling: per-channel normalize_latents=true requires "
                "training.scale_factor=1.0 for new runs. Legacy resume requires an "
                "explicit override plus checkpoint metadata proving normalization, "
                "the exact scale, and latent mean/std."
            )

    estimate = estimate_attention(config)
    safety = _get(config, "safety", {})
    max_tokens = int(_get(safety, "max_attention_tokens", 32768))
    max_attention_bytes = int(
        _get(safety, "max_training_attention_bytes", 64 * 1024 ** 3)
    )
    expert_override = bool(_get(safety, "allow_unsafe_global_attention", False))
    if (
        estimate["tokens"] > max_tokens
        or estimate["conservative_training_attention_bytes"] > max_attention_bytes
    ) and not expert_override:
        gib = estimate["conservative_training_attention_bytes"] / (1024 ** 3)
        raise ValueError(
            f"Infeasible global-attention grid has {estimate['tokens']:,} tokens "
            f"(conservative training attention estimate {gib:,.1f} GiB), above "
            f"a safety limit ({max_tokens:,} tokens or "
            f"{max_attention_bytes / (1024 ** 3):,.1f} GiB). Reduce the grid/increase patch size, or set "
            "safety.allow_unsafe_global_attention=true as an expert override."
        )
    return estimate

"""Versioned contract for self-describing precomputed Stage-1 latents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.utils.checkpointing import canonical_hash


LATENT_SCHEMA_NAME = "cardiodit.stage1-latent"
LATENT_SCHEMA_VERSION = 1


class LatentContractError(RuntimeError):
    """Raised when a latent is incomplete, stale, or internally inconsistent."""


_REQUIRED_CONTRACT_FIELDS = (
    "source",
    "shapes",
    "index_maps",
    "validity_masks",
    "preprocessing",
    "spacing",
    "quantized",
    "scale",
    "stage1",
)


def preprocessing_identity(contract: Mapping[str, Any]) -> str:
    preprocessing = contract.get("preprocessing")
    stage1 = contract.get("stage1") or {}
    return canonical_hash(
        {
            "schema_version": LATENT_SCHEMA_VERSION,
            "preprocessing": preprocessing,
            # Names/paths are useful provenance but not scientific identity;
            # identical bytes under a portable path must remain reusable.
            "stage1": {
                "config_sha256": stage1.get("config_sha256"),
                "checkpoint_sha256": stage1.get("checkpoint_sha256"),
            },
            "quantized": contract.get("quantized"),
            "scale": contract.get("scale"),
        }
    )


def reuse_identity(contract: Mapping[str, Any]) -> str:
    source = contract.get("source") or {}
    return canonical_hash(
        {
            "preprocessing_identity": preprocessing_identity(contract),
            "source_sha256": source.get("sha256"),
            "source_size": source.get("size"),
        }
    )


def make_latent_payload(latent: torch.Tensor, contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(contract)
    contract["preprocessing_identity"] = preprocessing_identity(contract)
    return {
        "latent_schema": {
            "name": LATENT_SCHEMA_NAME,
            "version": LATENT_SCHEMA_VERSION,
        },
        "latent": latent.detach().cpu(),
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
    }


def validate_latent_payload(
    payload: Any,
    *,
    expected_reuse_identity: str | None = None,
    allow_legacy: bool = False,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if isinstance(payload, torch.Tensor):
        if not allow_legacy:
            raise LatentContractError(
                "Legacy tensor-only latent has no preprocessing identity. "
                "Regenerate it or set the explicit allow_legacy_latents flag."
            )
        if not torch.isfinite(payload).all():
            raise LatentContractError("Legacy latent contains non-finite values")
        return payload, None
    if not isinstance(payload, Mapping):
        raise LatentContractError("Latent payload must be a mapping")
    schema = payload.get("latent_schema")
    if not isinstance(schema, Mapping) or schema.get("name") != LATENT_SCHEMA_NAME:
        raise LatentContractError(f"Unknown latent schema: {schema!r}")
    if schema.get("version") != LATENT_SCHEMA_VERSION:
        raise LatentContractError(f"Unsupported latent schema version: {schema!r}")
    latent = payload.get("latent")
    contract = payload.get("contract")
    if not isinstance(latent, torch.Tensor) or not isinstance(contract, Mapping):
        raise LatentContractError("Latent payload requires tensor 'latent' and mapping 'contract'")
    contract = dict(contract)
    missing = [field for field in _REQUIRED_CONTRACT_FIELDS if field not in contract]
    if missing:
        raise LatentContractError(f"Latent contract is incomplete; missing {missing}")
    observed_hash = payload.get("contract_sha256")
    expected_hash = canonical_hash(contract)
    if observed_hash != expected_hash:
        raise LatentContractError("Latent contract hash does not match its contents")
    if contract.get("preprocessing_identity") != preprocessing_identity(contract):
        raise LatentContractError("Latent preprocessing identity does not match its contract")
    latent_shape = tuple(int(value) for value in contract["shapes"].get("latent", ()))
    if tuple(latent.shape) != latent_shape:
        raise LatentContractError(
            f"Latent tensor shape {tuple(latent.shape)} does not match contract {latent_shape}"
        )
    if not torch.isfinite(latent).all():
        raise LatentContractError("Latent contains non-finite values")
    transformed_shape = tuple(
        int(value) for value in contract["shapes"].get("transformed_image", ())
    )
    if len(transformed_shape) != 5:
        raise LatentContractError(
            "Contract transformed_image shape must be (C,H,W,D,T)"
        )
    index_maps = contract["index_maps"]
    validity = contract["validity_masks"]
    if not isinstance(index_maps, Mapping) or not isinstance(validity, Mapping):
        raise LatentContractError("index_maps and validity_masks must be mappings")
    expected_lengths = {
        "temporal": transformed_shape[-1],
        "depth": transformed_shape[-2],
        "latent_temporal": latent_shape[-1],
        "latent_depth": latent_shape[-4],
    }
    for name, expected_length in expected_lengths.items():
        values = index_maps.get(name) if name in ("temporal", "depth") else validity.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != expected_length:
            raise LatentContractError(
                f"Contract {name} map/mask length does not match its tensor axis"
            )
    for name in ("temporal", "depth"):
        values = validity.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != expected_lengths[name]:
            raise LatentContractError(
                f"Contract {name} validity length does not match transformed image"
            )
    source = contract["source"]
    stage1 = contract["stage1"]
    scale = contract["scale"]
    if not isinstance(source, Mapping) or not source.get("sha256") or source.get("size") is None:
        raise LatentContractError("Contract source identity is incomplete")
    if not isinstance(stage1, Mapping) or not stage1.get("checkpoint_sha256") or not stage1.get("config_sha256"):
        raise LatentContractError("Contract Stage-1 identity is incomplete")
    if not isinstance(scale, Mapping) or "latent" not in scale:
        raise LatentContractError("Contract latent scale is incomplete")
    descriptor = contract.get("descriptor")
    if descriptor is not None:
        values = descriptor.get("latent_values") if isinstance(descriptor, Mapping) else None
        if not isinstance(values, (list, tuple)) or len(values) != latent_shape[-1]:
            raise LatentContractError(
                "Contract descriptor length does not match latent time"
            )
    if expected_reuse_identity is not None and reuse_identity(contract) != expected_reuse_identity:
        raise LatentContractError(
            "Existing latent identity does not match current source/preprocessing/Stage-1"
        )
    return latent, contract

"""Atomic sample publication and identity-bound completion manifests."""

from __future__ import annotations

import os
import platform
import tempfile
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from src.utils.checkpointing import atomic_json_save, canonical_hash, sha256_file


SAMPLE_SCHEMA = "cardiodit.sample-completion"
SAMPLE_SCHEMA_VERSION = 1
COMPLETION_MANIFEST = "completion_manifest.json"
FLAT_MANIFEST_DIR = ".manifests"


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def resolve_checkpoint_reference(path: str | Path) -> Path:
    """Resolve a modern checkpoint pointer without ever treating JSON as weights."""
    path = Path(path).expanduser().resolve()
    if path.suffix != ".json":
        return path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("pointer_schema") != {
        "name": "cardiodit.checkpoint-pointer",
        "version": 1,
    }:
        raise ValueError(f"Not a supported checkpoint pointer: {path}")
    target = (path.parent / payload["target"]).resolve()
    try:
        target.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"Checkpoint pointer escapes its run directory: {path}") from exc
    if not target.is_file() or target.suffix != ".pth":
        raise FileNotFoundError(f"Checkpoint pointer target is not a .pth file: {target}")
    return target


def software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import nibabel

        versions["nibabel"] = nibabel.__version__
    except ImportError:
        versions["nibabel"] = "unavailable"
    return versions


def per_sample_seeds(seed: int, sample_indices: Sequence[int]) -> list[int]:
    if seed is None:
        raise ValueError("A base seed is required for reproducible sampling.")
    return [int(seed) + int(index) for index in sample_indices]


def request_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    return {"sha256": canonical_hash(record), "payload": record}


def atomic_file_save(destination: str | Path, writer: Callable[[Path], None], suffix: str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        with tmp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def output_records(output_dir: str | Path, paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    records = []
    for value in paths:
        path = Path(value).resolve()
        records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return sorted(records, key=lambda item: item["path"])


def publish_completion_manifest(
    output_dir: str | Path,
    request: Mapping[str, Any],
    paths: Sequence[str | Path],
) -> Path:
    output_dir = Path(output_dir)
    records = output_records(output_dir, paths)
    expected_paths = {record["path"] for record in records}
    discovered_paths = {
        str(candidate.relative_to(output_dir.resolve()))
        for candidate in output_dir.resolve().rglob("*")
        if candidate.is_file() and candidate.name != COMPLETION_MANIFEST
    }
    extras = sorted(discovered_paths - expected_paths)
    if extras:
        raise RuntimeError(
            "Refusing to publish completion with unmanifested files; use a clean "
            f"identity-specific output directory. Extras: {extras}"
        )
    payload = {
        "schema": {"name": SAMPLE_SCHEMA, "version": SAMPLE_SCHEMA_VERSION},
        "request": request_record(request),
        "outputs": records,
        "outputs_sha256": canonical_hash(records),
        "status": "complete",
        "complete": True,
    }
    destination = output_dir / COMPLETION_MANIFEST
    atomic_json_save(payload, destination)
    return destination


def completion_matches(output_dir: str | Path, request: Mapping[str, Any]) -> bool:
    output_dir = Path(output_dir)
    path = output_dir / COMPLETION_MANIFEST
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != {
            "name": SAMPLE_SCHEMA,
            "version": SAMPLE_SCHEMA_VERSION,
        }:
            return False
        if payload.get("complete") is not True or payload.get("status") != "complete":
            return False
        stored_request = payload.get("request", {})
        stored_payload = stored_request.get("payload")
        stored_sha = stored_request.get("sha256")
        if not isinstance(stored_payload, dict) or stored_sha != canonical_hash(stored_payload):
            return False
        if stored_sha != canonical_hash(dict(request)):
            return False
        records = payload.get("outputs", [])
        if not isinstance(records, list) or payload.get("outputs_sha256") != canonical_hash(records):
            return False
        expected_paths = set()
        for record in records:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            output = output_dir / relative
            expected_paths.add(str(relative))
            if (
                not output.is_file()
                or output.stat().st_size != record.get("size")
                or sha256_file(output) != record.get("sha256")
            ):
                return False
        if len(expected_paths) != len(records):
            return False
        discovered_paths = {
            str(candidate.relative_to(output_dir))
            for candidate in output_dir.rglob("*")
            if candidate.is_file() and candidate != path
        }
        return bool(records) and discovered_paths == expected_paths
    except (OSError, ValueError, KeyError, TypeError):
        return False


def flat_manifest_path(output_dir: str | Path, sample_index: int) -> Path:
    return Path(output_dir) / FLAT_MANIFEST_DIR / f"sample_{int(sample_index):03d}.json"


def publish_flat_completion_manifest(
    output_dir: str | Path,
    sample_index: int,
    request: Mapping[str, Any],
    paths: Sequence[str | Path],
) -> Path:
    """Publish one manifest in a checkpoint directory shared by many samples."""
    output_dir = Path(output_dir)
    records = output_records(output_dir, paths)
    payload = {
        "schema": {"name": SAMPLE_SCHEMA, "version": SAMPLE_SCHEMA_VERSION},
        "request": request_record(request),
        "outputs": records,
        "outputs_sha256": canonical_hash(records),
        "status": "complete",
        "complete": True,
    }
    destination = flat_manifest_path(output_dir, sample_index)
    atomic_json_save(payload, destination)
    return destination


def flat_completion_matches(
    output_dir: str | Path,
    sample_index: int,
    request: Mapping[str, Any],
) -> bool:
    """Validate one sample without treating sibling flat samples as extras."""
    output_dir = Path(output_dir)
    path = flat_manifest_path(output_dir, sample_index)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != {
            "name": SAMPLE_SCHEMA,
            "version": SAMPLE_SCHEMA_VERSION,
        }:
            return False
        if payload.get("complete") is not True or payload.get("status") != "complete":
            return False
        stored_request = payload.get("request", {})
        stored_payload = stored_request.get("payload")
        stored_sha = stored_request.get("sha256")
        if not isinstance(stored_payload, dict) or stored_sha != canonical_hash(stored_payload):
            return False
        if stored_sha != canonical_hash(dict(request)):
            return False
        records = payload.get("outputs", [])
        if not isinstance(records, list) or payload.get("outputs_sha256") != canonical_hash(records):
            return False
        expected_paths = set()
        for record in records:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            output = output_dir / relative
            expected_paths.add(str(relative))
            if (
                not output.is_file()
                or output.stat().st_size != record.get("size")
                or sha256_file(output) != record.get("sha256")
            ):
                return False
        return bool(records) and len(expected_paths) == len(records)
    except (OSError, ValueError, KeyError, TypeError):
        return False

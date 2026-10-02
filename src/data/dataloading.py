# data/dataloading.py
import ast
import hashlib
import json
import math
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Sequence
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from monai.data import CacheDataset, PersistentDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandFlipd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
    MapTransform,
    Randomizable,
    Transform,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional
from src.data.temporal_alignment import (
    linear_phase,
    load_descriptor_sidecar,
    pool_alpha_t,
    resolve_descriptor_path,
)
from src.data.latent_contract import (
    LatentContractError,
    reuse_identity,
    validate_latent_payload,
)


# -------------------------
# Data dict helpers
# -------------------------

def get_datalist(ids_path: str, extended_report: bool = False) -> List[dict]:
    """Parse a CSV with an 'image' column into a list of MONAI data dicts."""
    df = pd.read_csv(ids_path, sep=",")
    base_dir = Path(ids_path).expanduser().resolve().parent
    data_dicts = [
        {"image": str(_resolve_manifest_path(row["image"], base_dir))}
        for _, row in df.iterrows()
    ]
    if extended_report:
        for i, d in enumerate(data_dicts[:5]):
            print(f"  {i}: {d}")
    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_latent_datalist(ids_path: str, extended_report: bool = False) -> List[dict]:
    """
    Parse a latent CSV while preserving optional sidecar metadata.

    Required column:
    - ``image``: path to a latent ``.pt`` tensor.

    Optional columns:
    - ``T_latent``: temporal latent length. If absent it is inferred from the
      tensor shape.
    - ``latent_shape`` or ``C_latent,Z_latent,X_latent,Y_latent,T_latent``:
      full latent tensor shape used for variable-shape bucket batching.
    - ``alpha_t`` / ``alpha_t_path`` / ``phase`` / ``phase_path``: descriptor
      sidecar path used for phase-aware positional encodings.
    """
    df = pd.read_csv(ids_path, sep=",")
    base_dir = Path(ids_path).expanduser().resolve().parent
    data_dicts: List[dict] = []
    for _, row in df.iterrows():
        d = {"image": str(_resolve_manifest_path(row["image"], base_dir))}
        if "latent_shape" in df.columns and pd.notna(row.get("latent_shape")):
            value = row["latent_shape"]
            if isinstance(value, str):
                parsed = ast.literal_eval(value) if value.strip().startswith(("[", "(")) else value.split(",")
            else:
                parsed = value
            d["latent_shape"] = tuple(int(v) for v in parsed)

        shape_cols = ["C_latent", "Z_latent", "X_latent", "Y_latent", "T_latent"]
        if all(col in df.columns for col in shape_cols):
            if all(pd.notna(row.get(col)) for col in shape_cols):
                d["latent_shape"] = tuple(int(row[col]) for col in shape_cols)

        if "T_latent" in df.columns and pd.notna(row.get("T_latent")):
            d["T_latent"] = int(row["T_latent"])
        for key in ("alpha_t", "alpha_t_path", "phase", "phase_path"):
            if key in df.columns and pd.notna(row.get(key)):
                d["alpha_t_path"] = str(_resolve_manifest_path(row[key], base_dir))
                break
        for key in (
            "preprocessing_identity",
            "latent_quantized",
            "latent_scale",
            "latent_schema_version",
            "stage1_checkpoint_sha256",
            "stage1_config_sha256",
            "source_sha256",
        ):
            if key in df.columns and pd.notna(row.get(key)):
                d[key] = row[key]
        data_dicts.append(d)

    if extended_report:
        for i, d in enumerate(data_dicts[:5]):
            print(f"  {i}: {d}")
    print(f"Found {len(data_dicts)} latent subjects.")
    return data_dicts


def _resolve_manifest_path(value, base_dir: Path) -> Path:
    """Resolve user/env paths relative to the manifest, not the process CWD."""
    import os

    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base_dir / expanded).resolve()


# -------------------------
# 4D-specific transforms
# -------------------------

class RandomZSliced(MapTransform, Randomizable):
    """
    Randomly sample one z-slice from a 4D CMR volume.

    Expects input shape (T, H, W, D) — MONAI loads NIfTI short-axis CMR as
    (H, W, D, T) and EnsureChannelFirst moves T to the front as the channel dim.
    Returns shape (1, H, W, T), ready for the VQ-GAN.

    Inherits from Randomizable so PersistentDataset/CacheDataset treat it as the
    pipeline split point: all deterministic transforms before this are cached,
    this transform and everything after are re-applied on every __getitem__.
    """

    def __init__(self, keys):
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self)

    def randomize(self, data=None):
        pass  # z-index drawn in __call__ since D is not known until data arrives

    def __call__(self, data: Dict):
        for key in self.keys:
            vol = data[key]          # (T, H, W, D)
            D = vol.shape[3]
            z = torch.randint(0, D, (1,)).item()
            sliced = vol[:, :, :, z]  # (T, H, W)
            data[key] = sliced.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, T)
        return data


class PermuteDimensionsd(MapTransform):
    """
    Permute the dimensions of a tensor in a MONAI data dict.

    ``perm`` indexes into the *full* tensor dimensions including the channel
    axis.  For a 5-D tensor (C, H, W, D, T) loaded from NIfTI, the default
    identity permutation ``(0, 1, 2, 3, 4)`` leaves the order unchanged.
    Adapt ``perm`` to match your data's on-disk axis ordering.

    Example — reorder from (C, T, H, W, D) to (C, H, W, D, T):
        PermuteDimensionsd(keys=["image"], perm=(0, 2, 3, 4, 1))
    """

    def __init__(self, keys, perm: tuple = (0, 1, 2, 3, 4)):
        super().__init__(keys)
        self.perm = perm

    def __call__(self, data: Dict):
        for key in self.keys:
            vol = data[key]
            if isinstance(vol, torch.Tensor):
                data[key] = vol.permute(*self.perm).contiguous()
            else:
                data[key] = np.transpose(vol, self.perm)
        return data


class UnsqueezeChanneld(MapTransform):
    """Insert a size-1 channel dim at position ``dim``."""

    def __init__(self, keys, dim: int = 0):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, data: Dict):
        for key in self.keys:
            vol = data[key]
            if isinstance(vol, torch.Tensor):
                data[key] = vol.unsqueeze(self.dim)
            else:
                data[key] = np.expand_dims(vol, self.dim)
        return data


class RandRotateHWTd(MapTransform, Randomizable):
    """Apply one in-plane H-W rotation consistently to every temporal frame.

    Inputs are expected to be channel-first ``(C, H, W, T)`` tensors.  Time is
    moved into the batch-like leading dimensions before the 2-D rotation, so
    interpolation can never combine values from different frames.
    """

    def __init__(
        self,
        keys,
        range_radians: float,
        prob: float = 0.1,
        mode: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self)
        self.range_radians = float(range_radians)
        self.prob = float(prob)
        if self.range_radians < 0:
            raise ValueError("range_radians must be non-negative")
        if not 0 <= self.prob <= 1:
            raise ValueError("prob must be in [0, 1]")
        self.mode = mode
        self._do_transform = False
        self._angle_degrees = 0.0

    def randomize(self, data=None):
        self._do_transform = bool(self.R.random() < self.prob)
        self._angle_degrees = float(
            self.R.uniform(-self.range_radians, self.range_radians) * 180.0 / math.pi
        )

    def __call__(self, data: Dict):
        self.randomize()
        if not self._do_transform:
            return data
        for key in self.keys:
            volume = data[key]
            was_tensor = isinstance(volume, torch.Tensor)
            tensor = volume if was_tensor else torch.as_tensor(volume)
            if tensor.ndim != 4:
                raise ValueError(
                    f"RandRotateHWTd expects (C,H,W,T), got {tuple(tensor.shape)}"
                )
            # torchvision rotates only the last two axes. Treat (C,T) as
            # leading dimensions and use the same sampled angle for all T.
            frames = tensor.permute(0, 3, 1, 2)
            rotated = tv_functional.rotate(
                frames,
                angle=self._angle_degrees,
                interpolation=self.mode,
                fill=0.0,
            ).permute(0, 2, 3, 1).contiguous()
            data[key] = rotated if was_tensor else rotated.cpu().numpy()
        return data


class CyclicPadTimed(MapTransform):
    """
    Cyclically repeat (then crop) the time axis of a tensor to ``target_frames``.

    ``dim`` selects which axis holds time. Default ``-1`` (last dim) matches the
    post-RandomZSliced shape (1, H, W, T). Pass ``dim=0`` to operate on the raw
    MONAI-loaded shape (T, H, W, D) so the transform can run before z-slicing.
    """

    def __init__(
        self,
        keys,
        target_frames: int,
        dim: int = -1,
        original_length_key: str = "original_temporal_length",
        index_map_key: str = "temporal_index_map",
        validity_key: str = "temporal_valid_mask",
        loss_mask_key: str = "loss_mask",
    ):
        super().__init__(keys)
        self.target_frames = int(target_frames)
        if self.target_frames <= 0:
            raise ValueError("target_frames must be a positive integer")
        self.dim = dim
        self.original_length_key = original_length_key
        self.index_map_key = index_map_key
        self.validity_key = validity_key
        self.loss_mask_key = loss_mask_key

    def __call__(self, data: Dict):
        for key in self.keys:
            volume = data[key]
            if not isinstance(volume, torch.Tensor):
                volume = torch.as_tensor(volume)

            dim = self.dim % volume.dim()
            T = volume.shape[dim]
            if T <= 0:
                raise ValueError("Cannot cyclically pad an empty temporal axis.")
            source_indices = torch.arange(self.target_frames, dtype=torch.long) % T
            valid = torch.arange(self.target_frames) < T
            if T < self.target_frames:
                n_repeats = (self.target_frames + T - 1) // T
                repeats = [1] * volume.dim()
                repeats[dim] = n_repeats
                volume = volume.repeat(*repeats)

            slices = [slice(None)] * volume.dim()
            slices[dim] = slice(None, self.target_frames)
            data[key] = volume[tuple(slices)]
            data[self.original_length_key] = torch.tensor(T, dtype=torch.long)
            data[self.index_map_key] = source_indices
            data[self.validity_key] = valid
            data[self.loss_mask_key] = valid.float()
        return data


class PadTimeToMultipleD(MapTransform):
    """
    Cyclically pad the time axis to the nearest multiple of ``multiple``.

    This preserves native temporal length when it is already divisible and only
    extends shorter remainders, e.g. 25 -> 28 for ``multiple=4``.
    """

    def __init__(
        self,
        keys,
        multiple: int,
        dim: int = -1,
        original_length_key: str = "original_temporal_length",
        index_map_key: str = "temporal_index_map",
        validity_key: str = "temporal_valid_mask",
        loss_mask_key: str = "loss_mask",
    ):
        super().__init__(keys)
        self.multiple = int(multiple)
        self.dim = dim
        self.original_length_key = original_length_key
        self.index_map_key = index_map_key
        self.validity_key = validity_key
        self.loss_mask_key = loss_mask_key
        if self.multiple <= 0:
            raise ValueError("multiple must be a positive integer")

    def __call__(self, data: Dict):
        for key in self.keys:
            volume = data[key]
            if not isinstance(volume, torch.Tensor):
                volume = torch.as_tensor(volume)

            dim = self.dim % volume.dim()
            T = int(volume.shape[dim])
            if T <= 0:
                raise ValueError("Cannot cyclically pad an empty temporal axis.")
            target_frames = int(math.ceil(T / self.multiple) * self.multiple)
            source_indices = torch.arange(target_frames, dtype=torch.long) % T
            valid = torch.arange(target_frames) < T
            if target_frames == T:
                data[key] = volume
                data[self.original_length_key] = torch.tensor(T, dtype=torch.long)
                data[self.index_map_key] = source_indices
                data[self.validity_key] = valid
                data[self.loss_mask_key] = valid.float()
                continue

            n_repeats = (target_frames + T - 1) // T
            repeats = [1] * volume.dim()
            repeats[dim] = n_repeats
            padded = volume.repeat(*repeats)

            slices = [slice(None)] * padded.dim()
            slices[dim] = slice(None, target_frames)
            data[key] = padded[tuple(slices)]
            data[self.original_length_key] = torch.tensor(T, dtype=torch.long)
            data[self.index_map_key] = source_indices
            data[self.validity_key] = valid
            data[self.loss_mask_key] = valid.float()
        return data


class MarkTemporalValidityd(MapTransform):
    """Attach identity temporal provenance when no padding/cropping is used."""

    def __init__(
        self,
        keys,
        dim: int = -1,
        original_length_key: str = "original_temporal_length",
        index_map_key: str = "temporal_index_map",
        validity_key: str = "temporal_valid_mask",
        loss_mask_key: str = "loss_mask",
    ):
        super().__init__(keys)
        self.dim = dim
        self.original_length_key = original_length_key
        self.index_map_key = index_map_key
        self.validity_key = validity_key
        self.loss_mask_key = loss_mask_key

    def __call__(self, data: Dict):
        lengths = {int(data[key].shape[self.dim % data[key].ndim]) for key in self.keys}
        if len(lengths) != 1:
            raise ValueError("all keys must share one temporal length")
        length = lengths.pop()
        valid = torch.ones(length, dtype=torch.bool)
        data[self.original_length_key] = torch.tensor(length, dtype=torch.long)
        data[self.index_map_key] = torch.arange(length, dtype=torch.long)
        data[self.validity_key] = valid
        data[self.loss_mask_key] = valid.float()
        return data


class RecordDepthProvenanced(MapTransform):
    """Record the exact source-depth map produced by centered crop/padding."""

    def __init__(
        self,
        keys,
        target_depth: int,
        dim: int = -2,
        index_map_key: str = "depth_index_map",
        validity_key: str = "depth_valid_mask",
        original_length_key: str = "original_depth",
    ):
        super().__init__(keys)
        self.target_depth = int(target_depth)
        self.dim = dim
        self.index_map_key = index_map_key
        self.validity_key = validity_key
        self.original_length_key = original_length_key
        if self.target_depth <= 0:
            raise ValueError("target_depth must be positive")

    def __call__(self, data: Dict):
        depths = {int(data[key].shape[self.dim % data[key].ndim]) for key in self.keys}
        if len(depths) != 1:
            raise ValueError("all keys must share one depth")
        depth = depths.pop()
        if depth >= self.target_depth:
            start = (depth - self.target_depth) // 2
            index_map = torch.arange(start, start + self.target_depth, dtype=torch.long)
            valid = torch.ones(self.target_depth, dtype=torch.bool)
        else:
            lower = (self.target_depth - depth) // 2
            index_map = torch.full((self.target_depth,), -1, dtype=torch.long)
            index_map[lower:lower + depth] = torch.arange(depth, dtype=torch.long)
            valid = index_map >= 0
        data[self.original_length_key] = torch.tensor(depth, dtype=torch.long)
        data[self.index_map_key] = index_map
        data[self.validity_key] = valid
        return data


# Bump whenever deterministic Stage-1 preprocessing semantics change.  The
# value is part of the on-disk cache directory, preventing stale reuse even if
# MONAI's transform serialization changes across releases.
VQGAN_TRANSFORM_VERSION = "stage1-preprocess-v2-hw-rotation-temporal-mask"


def vqgan_transform_fingerprint(
    roi_size: Sequence[int],
    target_frames: Optional[Union[int, str]],
    time_pad_multiple: Optional[int],
    spatial_permute: Optional[Sequence[int]],
) -> str:
    """Return a canonical fingerprint for every cache-affecting transform."""
    normalized_target = (
        "native"
        if target_frames is None or str(target_frames).lower() == "native"
        else int(target_frames)
    )
    contract = {
        "version": VQGAN_TRANSFORM_VERSION,
        "roi_size": [int(v) for v in roi_size],
        "target_frames": normalized_target,
        "time_pad_multiple": None if time_pad_multiple is None else int(time_pad_multiple),
        "spatial_permute": None if spatial_permute is None else [int(v) for v in spatial_permute],
        "intensity_range": [-1.0, 1.0],
        "temporal_policy": "cyclic_with_validity_mask",
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


# -------------------------
# Safe .pt loader
# -------------------------

def _safe_load(path):
    with torch.serialization.safe_globals([
        np.ndarray,
        np.dtype,
        np._core.multiarray._reconstruct,
    ]):
        return torch.load(path, weights_only=False)


def _padded_time_length(length: int, multiple: Optional[int]) -> int:
    if multiple is None or int(multiple) <= 1:
        return int(length)
    return int(math.ceil(int(length) / int(multiple)) * int(multiple))


def _cyclic_pad_tensor_to_length(tensor: torch.Tensor, target_length: int, dim: int = -1) -> torch.Tensor:
    dim = dim % tensor.dim()
    length = int(tensor.shape[dim])
    target_length = int(target_length)
    if target_length <= length:
        return tensor

    repeats = [1] * tensor.dim()
    repeats[dim] = int(math.ceil(target_length / length))
    padded = tensor.repeat(*repeats)
    slices = [slice(None)] * padded.dim()
    slices[dim] = slice(None, target_length)
    return padded[tuple(slices)]


def _time_loss_mask(original_length: int, padded_length: int) -> torch.Tensor:
    mask = torch.zeros(int(padded_length), dtype=torch.float32)
    mask[: int(original_length)] = 1.0
    return mask


# -------------------------
# Dataset for precomputed latents
# -------------------------

class LatentDataset(Dataset):
    """
    Loads precomputed 4D latent tensors from .pt files.

    Each file holds a tensor of shape (C, D, H, W, T) produced by
    ``src/scripts/encode_latents.py``.
    """

    def __init__(
        self,
        file_list: List,
        preload: bool = True,
        phase_dir: Optional[Union[str, Path]] = None,
        alpha_pool: Optional[str] = "linear",
        time_pad_multiple: Optional[int] = None,
        allow_legacy_latents: bool = False,
    ):
        self.entries = [
            f if isinstance(f, dict) else {"image": str(f)}
            for f in file_list
        ]
        self.file_list = [entry["image"] for entry in self.entries]
        self.preload = preload
        self.phase_dir = phase_dir
        self.alpha_pool = alpha_pool
        self.allow_legacy_latents = bool(allow_legacy_latents)
        self.time_pad_multiple = (
            int(time_pad_multiple) if time_pad_multiple is not None else None
        )
        if self.time_pad_multiple is not None and self.time_pad_multiple <= 0:
            raise ValueError("time_pad_multiple must be a positive integer")
        # Validate sequentially so lazy datasets never retain the full latent
        # corpus in a temporary list. Contracts and shapes are small metadata;
        # tensor storage is kept only when preload=True.
        tensors = [] if preload else None
        self.contracts = []
        original_shapes = []
        for entry, path in zip(self.entries, self.file_list):
            tensor, contract = validate_latent_payload(
                _safe_load(path), allow_legacy=self.allow_legacy_latents
            )
            shape = tuple(int(v) for v in tensor.shape)
            if "T_latent" in entry and int(entry["T_latent"]) != shape[-1]:
                raise LatentContractError("CSV T_latent disagrees with latent tensor")
            if "latent_shape" in entry and tuple(entry["latent_shape"]) != shape:
                raise LatentContractError("CSV latent_shape disagrees with latent tensor")
            self.contracts.append(contract)
            original_shapes.append(shape)
            if tensors is not None:
                tensors.append(tensor)
            del tensor
        self._reuse_identities = [
            None if contract is None else reuse_identity(contract)
            for contract in self.contracts
        ]
        has_legacy = [contract is None for contract in self.contracts]
        if any(has_legacy) and not all(has_legacy):
            raise LatentContractError(
                "A dataset cannot mix legacy tensor-only and contracted latents."
            )
        if self.contracts and self.contracts[0] is not None:
            identities = {contract["preprocessing_identity"] for contract in self.contracts}
            quantization_modes = {bool(contract["quantized"]) for contract in self.contracts}
            scales = {float(contract["scale"]["latent"]) for contract in self.contracts}
            if len(quantization_modes) != 1:
                raise LatentContractError(
                    "Latent dataset mixes quantized and continuous representations."
                )
            if len(scales) != 1:
                raise LatentContractError("Latent dataset contains mixed scale conventions.")
            if len(identities) != 1:
                raise LatentContractError(
                    "Latent dataset contains mixed preprocessing/Stage-1 identities."
                )
            for entry, contract in zip(self.entries, self.contracts):
                csv_identity = entry.get("preprocessing_identity")
                if csv_identity is not None and csv_identity != contract["preprocessing_identity"]:
                    raise LatentContractError(
                        "CSV preprocessing identity disagrees with latent payload."
                    )
        self.use_alpha = (
            phase_dir is not None
            or any("alpha_t_path" in entry for entry in self.entries)
            or any(contract is not None and "descriptor" in contract for contract in self.contracts)
        )

        self.data = tensors

        self._original_latent_shapes = original_shapes
        self._t_latent = [shape[-1] for shape in original_shapes]
        self._t_latent_padded = [
            _padded_time_length(t, self.time_pad_multiple) for t in self._t_latent
        ]
        self._latent_shapes = [
            (*shape[:-1], _padded_time_length(shape[-1], self.time_pad_multiple))
            for shape in self._original_latent_shapes
        ]

    def __len__(self):
        return len(self.file_list)

    def _infer_t_latent(self, idx: int, latent=None) -> int:
        entry = self.entries[idx]
        if latent is not None:
            return int(latent.shape[-1])
        if "T_latent" in entry:
            return int(entry["T_latent"])
        if "latent_shape" in entry:
            return int(entry["latent_shape"][-1])
        if latent is None:
            latent, _ = validate_latent_payload(
                _safe_load(entry["image"]), allow_legacy=self.allow_legacy_latents
            )
        return int(latent.shape[-1])

    def _infer_latent_shape(self, idx: int, latent=None) -> Tuple[int, ...]:
        entry = self.entries[idx]
        if latent is not None:
            return tuple(int(v) for v in latent.shape)
        if "latent_shape" in entry:
            return tuple(int(v) for v in entry["latent_shape"])
        if latent is None:
            latent, _ = validate_latent_payload(
                _safe_load(entry["image"]), allow_legacy=self.allow_legacy_latents
            )
        return tuple(int(v) for v in latent.shape)

    def _load_alpha_t(self, idx: int, t_latent: int) -> torch.Tensor:
        entry = self.entries[idx]
        contract = self.contracts[idx]
        if contract is not None and "descriptor" in contract:
            latent_values = contract["descriptor"].get("latent_values")
            if latent_values is not None:
                return pool_alpha_t(latent_values, t_latent, method=self.alpha_pool)
        path = entry.get("alpha_t_path")
        if path is None:
            resolved = resolve_descriptor_path(entry["image"], self.phase_dir)
            path = str(resolved) if resolved is not None else None

        if path is None:
            alpha_t = linear_phase(t_latent)
        else:
            alpha_t = load_descriptor_sidecar(path)
        return pool_alpha_t(alpha_t, t_latent, method=self.alpha_pool)

    def __getitem__(self, idx):
        if self.preload:
            latent = self.data[idx]
        else:
            latent, contract = validate_latent_payload(
                _safe_load(self.file_list[idx]),
                allow_legacy=self.allow_legacy_latents,
            )
            expected_identity = self._reuse_identities[idx]
            if expected_identity is not None and reuse_identity(contract) != expected_identity:
                raise LatentContractError("Latent identity changed after dataset construction")

        t_latent = int(latent.shape[-1])
        t_padded = _padded_time_length(t_latent, self.time_pad_multiple)
        latent = _cyclic_pad_tensor_to_length(latent, t_padded, dim=-1)
        item = {
            "image": latent,
            "T_latent": torch.tensor(t_latent, dtype=torch.long),
            "T_latent_padded": torch.tensor(t_padded, dtype=torch.long),
            "latent_shape": torch.tensor((*tuple(latent.shape[:-1]), t_padded), dtype=torch.long),
        }
        contract = self.contracts[idx]
        if contract is None:
            loss_mask = torch.ones(t_latent, dtype=torch.float32)
        else:
            loss_mask = torch.as_tensor(
                contract["validity_masks"]["latent_temporal"], dtype=torch.float32
            )
            if loss_mask.numel() != t_latent:
                raise LatentContractError(
                    "Latent temporal validity length does not match latent tensor"
                )
        if t_padded > t_latent:
            loss_mask = torch.cat(
                [loss_mask, torch.zeros(t_padded - t_latent, dtype=torch.float32)]
            )
        item["loss_mask"] = loss_mask
        if self.use_alpha:
            alpha_t = self._load_alpha_t(idx, t_latent)
            item["alpha_t"] = _cyclic_pad_tensor_to_length(alpha_t, t_padded, dim=-1)
        return item


def _shape_key(shape: Union[int, Sequence[int], torch.Tensor]) -> Tuple[int, ...]:
    if isinstance(shape, torch.Tensor):
        shape = shape.detach().cpu().tolist()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(v) for v in shape)


class ShapeBucketBatchSampler(Sampler[List[int]]):
    """Batch latent indices with the same full latent shape to avoid ragged tensors."""

    def __init__(
        self,
        shapes: Sequence[Union[int, Sequence[int], torch.Tensor]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.shapes = [_shape_key(v) for v in shapes]
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        for idx, shape in enumerate(self.shapes):
            buckets.setdefault(shape, []).append(idx)

        keys = list(buckets)
        if self.shuffle:
            order = torch.randperm(len(keys), generator=generator).tolist()
            keys = [keys[i] for i in order]

        for key in keys:
            indices = buckets[key]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        buckets: Dict[Tuple[int, ...], int] = {}
        for shape in self.shapes:
            buckets[shape] = buckets.get(shape, 0) + 1
        for count in buckets.values():
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total


class DistributedShapeBucketBatchSampler(Sampler[List[int]]):
    """DDP-safe same-shape bucket sampler with equal batch counts per rank."""

    def __init__(
        self,
        shapes: Sequence[Union[int, Sequence[int], torch.Tensor]],
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        pad_to_equal_batches: bool = True,
    ):
        self.shapes = [_shape_key(v) for v in shapes]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0
        self.pad_to_equal_batches = bool(pad_to_equal_batches)
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas})")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _bucket_counts(self) -> Dict[Tuple[int, ...], int]:
        buckets: Dict[Tuple[int, ...], int] = {}
        for shape in self.shapes:
            buckets[shape] = buckets.get(shape, 0) + 1
        return buckets

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        for idx, shape in enumerate(self.shapes):
            buckets.setdefault(shape, []).append(idx)

        keys = list(buckets)
        if self.shuffle:
            order = torch.randperm(len(keys), generator=generator).tolist()
            keys = [keys[i] for i in order]

        global_batch = self.batch_size * self.num_replicas
        for key in keys:
            indices = buckets[key]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]

            if not self.pad_to_equal_batches:
                rank_indices = indices[self.rank::self.num_replicas]
                for start in range(0, len(rank_indices), self.batch_size):
                    batch = rank_indices[start:start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        yield batch
                continue
            if self.drop_last:
                total = (len(indices) // global_batch) * global_batch
                indices = indices[:total]
            else:
                total = ((len(indices) + global_batch - 1) // global_batch) * global_batch
                if total > len(indices):
                    pad = total - len(indices)
                    repeats = (pad + len(indices) - 1) // len(indices)
                    indices = indices + (indices * repeats)[:pad]

            rank_indices = indices[self.rank:total:self.num_replicas]
            for start in range(0, len(rank_indices), self.batch_size):
                batch = rank_indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        global_batch = self.batch_size * self.num_replicas
        for count in self._bucket_counts().values():
            if not self.pad_to_equal_batches:
                local_count = max(
                    0,
                    (count + self.num_replicas - 1 - self.rank)
                    // self.num_replicas,
                )
                if self.drop_last:
                    total += local_count // self.batch_size
                else:
                    total += (local_count + self.batch_size - 1) // self.batch_size
            elif self.drop_last:
                total += count // global_batch
            else:
                total += (count + global_batch - 1) // global_batch
        return total

    def padding_stats(self) -> Dict[str, Union[int, float]]:
        """Quantify the minimal per-shape padding needed for lock-step DDP."""
        source = sum(self._bucket_counts().values())
        if not self.pad_to_equal_batches or self.drop_last:
            padded = source
        else:
            global_batch = self.batch_size * self.num_replicas
            padded = sum(
                ((count + global_batch - 1) // global_batch) * global_batch
                for count in self._bucket_counts().values()
            )
        padding = padded - source
        return {
            "source_examples": source,
            "padded_examples": padded,
            "padding_examples": padding,
            "padding_fraction": (padding / padded) if padded else 0.0,
        }


class DistributedEvalSampler(Sampler[int]):
    """Deterministic distributed sharding with no duplicate padding."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas})")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return max(
            0,
            (len(self.dataset) + self.num_replicas - 1 - self.rank)
            // self.num_replicas,
        )


class TemporalBucketBatchSampler(ShapeBucketBatchSampler):
    """Backward-compatible alias; pass full shapes for variable-shape training."""


# -------------------------
# Dataloader for VQ-GAN training (raw CMR slices)
# -------------------------

def get_vqgan_dataloader(
    training_ids: str,
    validation_ids: str,
    batch_size: int,
    num_workers: int = 8,
    rank: int = 0,
    world_size: int = 1,
    roi_size: Tuple[int, int, int] = (224, 224, 32),
    target_frames: Optional[Union[int, str]] = 32,
    time_pad_multiple: Optional[int] = None,
    use_persistent: bool = False,
    cache_dir: Union[str, Path] = "/tmp/vqgan_cache",
    spatial_permute: Optional[Tuple[int, ...]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for VQ-GAN training on 2D+t (H, W, T) CMR slices.

    Input CSV must point to full 4D NIfTI volumes. After LoadImaged +
    EnsureChannelFirstd the expected shape is (T, H, W, D). If the dataset
    stores axes in a different order (e.g. MNM2 is (D, H, W, T) on disk →
    EnsureChannelFirst gives (T, D, H, W)), pass ``spatial_permute`` to
    reorder: e.g. (0, 2, 3, 1) maps (T,D,H,W) → (T,H,W,D).
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    # Deterministic transforms — cached to disk by PersistentDataset (or RAM by
    # CacheDataset). Operate on the full (T, H, W, D) volume. CenterSpatialCropd
    # and SpatialPadd use -1 for the D axis so z-slices are left untouched.
    roi_xy = (roi_size[0], roi_size[1], -1)
    det_transforms_list = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),                                   # → (T, H, W, D) if UMM-ordered
    ]
    if spatial_permute is not None:
        det_transforms_list.append(
            PermuteDimensionsd(keys=["image"], perm=spatial_permute)           # fix axis order → (T, H, W, D)
        )
    if isinstance(target_frames, str):
        target_frames = None if target_frames.lower() == "native" else int(target_frames)
    if target_frames is not None:
        target_frames = int(target_frames)
    if target_frames is not None and time_pad_multiple is not None:
        raise ValueError("Set either target_frames or time_pad_multiple, not both.")

    det_transforms_list.append(ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0))
    if target_frames is not None:
        det_transforms_list.append(
            CyclicPadTimed(keys=["image"], target_frames=target_frames, dim=0)  # → (target_frames, H, W, D)
        )
    elif time_pad_multiple is not None:
        det_transforms_list.append(
            PadTimeToMultipleD(keys=["image"], multiple=int(time_pad_multiple), dim=0)
        )
    else:
        det_transforms_list.append(MarkTemporalValidityd(keys=["image"], dim=0))
    det_transforms_list.extend([
        CenterSpatialCropd(keys=["image"], roi_size=roi_xy),                   # → (T, roi_h, roi_w, D)
        SpatialPadd(keys=["image"], spatial_size=roi_xy, constant_values=-1.0),
    ])
    det_transforms = Compose(det_transforms_list)

    # Random transforms — re-applied on every __getitem__. PersistentDataset
    # recognises RandomZSliced as Randomizable and splits the pipeline here.
    train_rand_transforms = Compose([
        RandomZSliced(keys=["image"]),                     # (T, H, W, D) → (1, H, W, T)
        RandRotateHWTd(keys=["image"], range_radians=0.0872665, prob=0.2),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        ToTensord(keys=["image"]),
    ])

    val_rand_transforms = Compose([
        RandomZSliced(keys=["image"]),                     # (T, H, W, D) → (1, H, W, T)
        ToTensord(keys=["image"]),
    ])

    train_transforms = Compose([*det_transforms.transforms, *train_rand_transforms.transforms])
    val_transforms = Compose([*det_transforms.transforms, *val_rand_transforms.transforms])

    if use_persistent:
        transform_id = vqgan_transform_fingerprint(
            roi_size=roi_size,
            target_frames=target_frames,
            time_pad_multiple=time_pad_multiple,
            spatial_permute=spatial_permute,
        )
        versioned_cache_dir = Path(cache_dir) / f"transform_{transform_id}"
        train_ds = PersistentDataset(
            data=train_files, transform=train_transforms,
            cache_dir=str(versioned_cache_dir / "train"),
        )
        val_ds = PersistentDataset(
            data=val_files, transform=val_transforms,
            cache_dir=str(versioned_cache_dir / "val"),
        )
    else:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedEvalSampler(val_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# -------------------------
# Dataloader for DiT training (precomputed 4D latents)
# -------------------------

def get_dit_dataloader(
    training_ids: str,
    validation_ids: str,
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    use_precomputed_latents: bool = True,
    preload_latents: bool = True,
    phase_dir: Optional[Union[str, Path]] = None,
    use_bucket_sampler: bool = False,
    alpha_pool: Optional[str] = "linear",
    batch_strategy: Optional[str] = None,
    variable_shape: bool = False,
    latent_time_pad_multiple: Optional[int] = None,
    allow_legacy_latents: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for DiT training on precomputed 4D latents.

    Each .pt file holds a (C, D, H, W, T) tensor produced by encode_latents.py.
    """
    train_files = get_latent_datalist(training_ids)
    val_files = get_latent_datalist(validation_ids)

    if not use_precomputed_latents:
        raise NotImplementedError(
            "On-the-fly 4D encoding is not supported here. "
            "Pre-encode with src/scripts/encode_latents.py first."
        )

    train_ds = LatentDataset(
        train_files, preload=preload_latents,
        phase_dir=phase_dir, alpha_pool=alpha_pool,
        time_pad_multiple=latent_time_pad_multiple,
        allow_legacy_latents=allow_legacy_latents,
    )
    val_ds = LatentDataset(
        val_files, preload=False,
        phase_dir=phase_dir, alpha_pool=alpha_pool,
        time_pad_multiple=latent_time_pad_multiple,
        allow_legacy_latents=allow_legacy_latents,
    )

    if batch_strategy is None:
        batch_strategy = "bucket" if use_bucket_sampler else "fixed"
    if use_bucket_sampler:
        batch_strategy = "bucket"
    batch_strategy = str(batch_strategy).lower()
    if batch_strategy == "microbatch":
        batch_strategy = "grad_accum_microbatch"
    if batch_strategy not in {"fixed", "bucket", "grad_accum_microbatch", "mask"}:
        raise ValueError(
            "training.batch_strategy must be fixed, bucket, grad_accum_microbatch, or mask; "
            f"got {batch_strategy!r}"
        )
    if batch_strategy == "mask":
        raise NotImplementedError(
            "Masked variable-shape packing is not implemented. Use batch_strategy=bucket "
            "or grad_accum_microbatch."
        )

    all_shapes = set(train_ds._latent_shapes) | set(val_ds._latent_shapes)
    has_variable_shapes = len(all_shapes) > 1
    if has_variable_shapes and batch_strategy == "fixed" and batch_size > 1:
        raise ValueError(
            "Latent CSV contains variable full shapes but training.batch_strategy=fixed "
            f"with batch_size={batch_size}. Use batch_strategy=bucket or "
            "grad_accum_microbatch, or pre-pad/crop latents to one shape."
        )
    if variable_shape and batch_strategy == "fixed" and batch_size > 1 and has_variable_shapes:
        raise ValueError(
            "variable_shape=true requires shape-aware batching when batch_size > 1."
        )

    effective_batch_size = 1 if batch_strategy == "grad_accum_microbatch" else batch_size

    train_sampler = None
    val_sampler = None
    train_batch_sampler = None
    val_batch_sampler = None

    if batch_strategy == "bucket":
        if world_size > 1:
            train_batch_sampler = DistributedShapeBucketBatchSampler(
                train_ds._latent_shapes, batch_size=effective_batch_size,
                num_replicas=world_size, rank=rank, shuffle=True,
            )
            val_batch_sampler = DistributedShapeBucketBatchSampler(
                val_ds._latent_shapes, batch_size=effective_batch_size,
                num_replicas=world_size, rank=rank, shuffle=False,
                pad_to_equal_batches=False,
            )
        else:
            train_batch_sampler = ShapeBucketBatchSampler(
                train_ds._latent_shapes, batch_size=effective_batch_size, shuffle=True,
            )
            val_batch_sampler = ShapeBucketBatchSampler(
                val_ds._latent_shapes, batch_size=effective_batch_size, shuffle=False,
            )
    elif world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedEvalSampler(val_ds, num_replicas=world_size, rank=rank)

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=effective_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    if val_batch_sampler is not None:
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=effective_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader

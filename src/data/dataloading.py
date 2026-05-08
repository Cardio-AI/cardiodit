# data/dataloading.py
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List, Dict
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from monai.data import CacheDataset, PersistentDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated,
    RandFlipd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
    MapTransform,
    Randomizable,
)


# -------------------------
# Data dict helpers
# -------------------------

def get_datalist(ids_path: str, extended_report: bool = False) -> List[dict]:
    """Parse a CSV with an 'image' column into a list of MONAI data dicts."""
    df = pd.read_csv(ids_path, sep=",")
    data_dicts = [{"image": str(row["image"])} for _, row in df.iterrows()]
    if extended_report:
        for i, d in enumerate(data_dicts[:5]):
            print(f"  {i}: {d}")
    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


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


class CyclicPadTimed(MapTransform):
    """
    Cyclically repeat (then crop) the time axis of a tensor to ``target_frames``.

    ``dim`` selects which axis holds time. Default ``-1`` (last dim) matches the
    post-RandomZSliced shape (1, H, W, T). Pass ``dim=0`` to operate on the raw
    MONAI-loaded shape (T, H, W, D) so the transform can run before z-slicing.
    """

    def __init__(self, keys, target_frames: int, dim: int = -1):
        super().__init__(keys)
        self.target_frames = target_frames
        self.dim = dim

    def __call__(self, data: Dict):
        for key in self.keys:
            volume = data[key]
            if not isinstance(volume, torch.Tensor):
                volume = torch.as_tensor(volume)

            dim = self.dim % volume.dim()
            T = volume.shape[dim]
            if T < self.target_frames:
                n_repeats = (self.target_frames + T - 1) // T
                repeats = [1] * volume.dim()
                repeats[dim] = n_repeats
                volume = volume.repeat(*repeats)

            slices = [slice(None)] * volume.dim()
            slices[dim] = slice(None, self.target_frames)
            data[key] = volume[tuple(slices)]
        return data


# -------------------------
# Safe .pt loader
# -------------------------

def _safe_load(path):
    reconstruct = getattr(
        getattr(np, "_core", np.core).multiarray,
        "_reconstruct",
    )
    with torch.serialization.safe_globals([np.ndarray, np.dtype, reconstruct]):
        return torch.load(path, weights_only=False)


# -------------------------
# Dataset for precomputed latents
# -------------------------

class LatentDataset(Dataset):
    """
    Loads precomputed 4D latent tensors from .pt files.

    Each file holds a tensor of shape (C, D, H, W, T) produced by
    ``src/scripts/encode_latents.py``.
    """

    def __init__(self, file_list: List, preload: bool = True):
        self.file_list = [
            f["image"] if isinstance(f, dict) else f
            for f in file_list
        ]
        self.preload = preload

        if preload:
            self.data = [_safe_load(f) for f in self.file_list]
        else:
            self.data = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.preload:
            return {"image": self.data[idx]}
        return {"image": _safe_load(self.file_list[idx])}


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
    target_frames: int = 32,
    use_persistent: bool = False,
    cache_dir: Union[str, Path] = "/tmp/vqgan_cache",
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for VQ-GAN training on 2D+t (H, W, T) CMR slices.

    Input CSV must point to full 4D NIfTI volumes (H, W, D, T). MONAI loads
    them as (1, H, W, D, T). ``RandomZSliced`` then draws one random z-slice
    per sample to produce (1, H, W, T) for the VQ-GAN. The VQ-GAN is
    z-position-agnostic by design — it is a universal 2D+t codec; all 4D
    spatial relations are the DiT's responsibility.
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    # Deterministic transforms — cached to disk by PersistentDataset (or RAM by
    # CacheDataset). Operate on the full (T, H, W, D) volume. CenterSpatialCropd
    # and SpatialPadd use -1 for the D axis so z-slices are left untouched.
    roi_xy = (roi_size[0], roi_size[1], -1)
    det_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),                                   # → (T, H, W, D)
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        CyclicPadTimed(keys=["image"], target_frames=target_frames, dim=0),    # → (target_frames, H, W, D)
        CenterSpatialCropd(keys=["image"], roi_size=roi_xy),                   # → (target_frames, roi_h, roi_w, D)
        SpatialPadd(keys=["image"], spatial_size=roi_xy,constant_values=-1.0),
    ])

    # Random transforms — re-applied on every __getitem__. PersistentDataset
    # recognises RandomZSliced as Randomizable and splits the pipeline here.
    train_rand_transforms = Compose([
        RandomZSliced(keys=["image"]),                     # (T, H, W, D) → (1, H, W, T)
        RandRotated(keys=["image"], range_x=0.0872665, prob=0.2),
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
        train_ds = PersistentDataset(
            data=train_files, transform=train_transforms,
            cache_dir=str(Path(cache_dir) / "train"),
        )
        val_ds = PersistentDataset(
            data=val_files, transform=val_transforms,
            cache_dir=str(Path(cache_dir) / "val"),
        )
    else:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = None

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
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for DiT training on precomputed 4D latents.

    Each .pt file holds a (C, D, H, W, T) tensor produced by encode_latents.py.
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    if not use_precomputed_latents:
        raise NotImplementedError(
            "On-the-fly 4D encoding is not supported here. "
            "Pre-encode with src/scripts/encode_latents.py first."
        )

    train_ds = LatentDataset(train_files, preload=preload_latents)
    val_ds = LatentDataset(val_files, preload=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = None

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
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

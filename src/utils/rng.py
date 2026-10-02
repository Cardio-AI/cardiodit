"""Scoped deterministic RNG streams that cannot perturb training randomness."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


@contextmanager
def isolated_rng(seed: int, device: torch.device | str | None = None) -> Iterator[None]:
    """Run with a fixed RNG stream, restoring every caller-visible state.

    ``torch.random.fork_rng`` preserves CPU and the selected CUDA device. Python
    and NumPy require explicit snapshots. This is suitable for validation and
    sample logging inside a training process.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device = torch.device(device) if device is not None else torch.device("cpu")
    cuda_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(int(seed))
            np.random.seed(int(seed) % (2 ** 32))
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

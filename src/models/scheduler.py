from __future__ import annotations

from collections import namedtuple
from keyword import iskeyword
from textwrap import dedent, indent
from typing import Callable, Dict, Iterable, TypeVar

import torch
import torch.nn as nn

from src.models.utils import unsqueeze_right

T = TypeVar("T")


def _is_variable(name):
    return name.isidentifier() and not iskeyword(name)


class ComponentStore:
    _Component = namedtuple("Component", ("description", "value"))

    def __init__(self, name: str, description: str) -> None:
        self.components: Dict[str, self._Component] = {}
        self.name = name
        self.description = description

    def add(self, name: str, desc: str, value: T) -> T:
        if not _is_variable(name):
            raise ValueError("Component name must be a valid Python identifier")
        self.components[name] = self._Component(desc, value)
        return value

    def add_def(self, name: str, desc: str) -> Callable:
        def deco(func):
            return self.add(name, desc, func)

        return deco

    def __iter__(self) -> Iterable:
        for key, value in self.components.items():
            yield key, value.value

    def __str__(self):
        result = f"Component Store '{self.name}': {self.description}\nAvailable components:"
        for key, value in self.components.items():
            result += f"\n* {key}:"
            if hasattr(value.value, "__doc__"):
                doc = indent(dedent(value.value.__doc__.lstrip("\n").rstrip()), "    ")
                result += f"\n{doc}\n"
            else:
                result += f" {value.description}"
        return result

    def __getitem__(self, name: str):
        if name not in self.components:
            raise ValueError(f"Component '{name}' not found")
        return self.components[name].value


NoiseSchedules = ComponentStore("NoiseSchedules", "Functions to generate noise schedules")


@NoiseSchedules.add_def("linear_beta", "Linear beta schedule")
def _linear_beta(
    num_train_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
):
    return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)


@NoiseSchedules.add_def("scaled_linear_beta", "Scaled linear beta schedule")
def _scaled_linear_beta(
    num_train_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
):
    return (
        torch.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_train_timesteps,
            dtype=torch.float32,
        )
        ** 2
    )


@NoiseSchedules.add_def("sigmoid_beta", "Sigmoid beta schedule")
def _sigmoid_beta(
    num_train_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    sig_range: float = 6,
):
    betas = torch.linspace(-sig_range, sig_range, num_train_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


@NoiseSchedules.add_def("cosine", "Cosine beta schedule")
def _cosine_beta(num_train_timesteps: int, s: float = 8e-3):
    x = torch.linspace(
        0,
        num_train_timesteps,
        num_train_timesteps + 1,
        dtype=torch.float64,
    )
    alphas_cumprod = torch.cos(
        ((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5
    ) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999).float()


class Scheduler(nn.Module):
    """Base class for beta-schedule diffusion schedulers."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "linear_beta",
        **schedule_args,
    ) -> None:
        super().__init__()
        schedule_args["num_train_timesteps"] = num_train_timesteps
        noise_sched = NoiseSchedules[schedule](**schedule_args)

        if isinstance(noise_sched, tuple):
            self.betas, self.alphas, self.alphas_cumprod = noise_sched
        else:
            self.betas = noise_sched
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.num_train_timesteps = num_train_timesteps
        self.one = torch.tensor(1.0)
        self.num_inference_steps = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_cumprod = unsqueeze_right(
            self.alphas_cumprod[timesteps] ** 0.5, original_samples.ndim
        )
        sqrt_one_minus_alpha_prod = unsqueeze_right(
            (1 - self.alphas_cumprod[timesteps]) ** 0.5, original_samples.ndim
        )
        return sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_prod * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = unsqueeze_right(
            self.alphas_cumprod[timesteps] ** 0.5, sample.ndim
        )
        sqrt_one_minus_alpha_prod = unsqueeze_right(
            (1 - self.alphas_cumprod[timesteps]) ** 0.5, sample.ndim
        )
        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from monai.utils import StrEnum

from .scheduler import Scheduler


def _compatible_generator(
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Generator | None:
    """Use a generator only when PyTorch can draw on the target device."""
    if generator is None:
        return None
    generator_device = getattr(generator, "device", None)
    if generator_device is None:
        return generator
    generator_device = torch.device(generator_device)
    device = torch.device(device)
    if generator_device.type != device.type:
        return None
    if (
        device.type == "cuda"
        and generator_device.index is not None
        and device.index is not None
        and generator_device.index != device.index
    ):
        return None
    return generator


class DDPMVarianceType(StrEnum):
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"


class DDPMPredictionType(StrEnum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class DDPMScheduler(Scheduler):
    """
    DDPM scheduler with cosine/linear noise schedule and v-prediction support.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "linear_beta",
        variance_type: str = DDPMVarianceType.FIXED_SMALL,
        clip_sample: bool = False,
        prediction_type: str = DDPMPredictionType.EPSILON,
        **schedule_args,
    ) -> None:
        super().__init__(num_train_timesteps, schedule, **schedule_args)

        if variance_type not in DDPMVarianceType.__members__.values():
            raise ValueError("Argument `variance_type` must be a member of `DDPMVarianceType`")

        if prediction_type not in DDPMPredictionType.__members__.values():
            raise ValueError("Argument `prediction_type` must be a member of `DDPMPredictionType`")

        self.clip_sample = clip_sample
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.solver_order = 1

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be at least 1.")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps}."
            )

        self.num_inference_steps = num_inference_steps
        if num_inference_steps == 1:
            timesteps = torch.tensor([self.num_train_timesteps - 1], dtype=torch.long)
        else:
            timesteps = torch.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.float64,
            ).round().to(torch.long)
        if timesteps.numel() > 1 and not torch.all(timesteps[:-1] > timesteps[1:]):
            raise RuntimeError("DDPM inference timestep grid must be strictly decreasing.")
        self.timesteps = timesteps.to(device)

    def _get_previous_timestep(self, timestep: int) -> int:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps() before step().")
        matches = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if matches.numel() != 1:
            raise ValueError(
                f"Timestep {timestep} is not a unique member of the configured inference grid."
            )
        index = int(matches.item())
        return int(self.timesteps[index + 1]) if index + 1 < len(self.timesteps) else -1

    def _transition_terms(
        self,
        timestep: int,
        previous_timestep: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if previous_timestep >= timestep:
            raise ValueError(
                "DDPM denoising requires previous_timestep < timestep; "
                f"got {previous_timestep} >= {timestep}."
            )
        schedule_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        alpha_prod_t = self.alphas_cumprod[timestep].to(device=device, dtype=schedule_dtype)
        alpha_prod_s = (
            self.alphas_cumprod[previous_timestep].to(device=device, dtype=schedule_dtype)
            if previous_timestep >= 0
            else torch.ones((), device=device, dtype=schedule_dtype)
        )
        alpha_t_given_s = alpha_prod_t / alpha_prod_s
        beta_t_given_s = 1 - alpha_t_given_s
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_s = 1 - alpha_prod_s
        if not bool(torch.isfinite(alpha_t_given_s)) or not bool(alpha_t_given_s >= 0):
            raise RuntimeError("DDPM transition has a non-finite or negative cumulative alpha ratio.")
        if not bool(beta_t_given_s > 0) or not bool(beta_prod_t > 0):
            raise RuntimeError("DDPM transition has a zero or negative noise ratio.")
        return alpha_prod_t, alpha_prod_s, alpha_t_given_s, beta_prod_t, beta_prod_s

    def _get_mean(
        self,
        timestep: int,
        previous_timestep: int,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        _, alpha_prod_s, alpha_t_given_s, beta_prod_t, beta_prod_s = self._transition_terms(
            timestep,
            previous_timestep,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        beta_t_given_s = 1 - alpha_t_given_s
        x_0_coefficient = alpha_prod_s.sqrt() * beta_t_given_s / beta_prod_t
        x_t_coefficient = alpha_t_given_s.sqrt() * beta_prod_s / beta_prod_t

        return x_0_coefficient * x_0 + x_t_coefficient * x_t

    def _get_variance(
        self,
        timestep: int,
        previous_timestep: int,
        predicted_variance: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self.alphas_cumprod.device
        dtype = dtype or self.alphas_cumprod.dtype
        _, _, alpha_t_given_s, beta_prod_t, beta_prod_s = self._transition_terms(
            timestep,
            previous_timestep,
            device=device,
            dtype=dtype,
        )
        beta_t_given_s = 1 - alpha_t_given_s
        variance = beta_prod_s / beta_prod_t * beta_t_given_s
        if self.variance_type == DDPMVarianceType.FIXED_SMALL:
            variance = torch.clamp(variance, min=1e-20)
        elif self.variance_type == DDPMVarianceType.FIXED_LARGE:
            variance = beta_t_given_s
        elif self.variance_type == DDPMVarianceType.LEARNED:
            if predicted_variance is None:
                raise ValueError("Learned DDPM variance requires a predicted variance tensor.")
            return predicted_variance
        elif self.variance_type == DDPMVarianceType.LEARNED_RANGE:
            if predicted_variance is None:
                raise ValueError("Learned-range DDPM variance requires a predicted variance tensor.")
            min_log = variance
            max_log = beta_t_given_s
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = int(timestep)
        previous_timestep = self._get_previous_timestep(timestep)
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        schedule_dtype = (
            torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        )
        alpha_prod_t = self.alphas_cumprod[timestep].to(
            device=sample.device,
            dtype=schedule_dtype,
        )
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == DDPMPredictionType.EPSILON:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == DDPMPredictionType.SAMPLE:
            pred_original_sample = model_output
        elif self.prediction_type == DDPMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        if previous_timestep == -1:
            return pred_original_sample, pred_original_sample

        pred_prev_sample = self._get_mean(
            timestep,
            previous_timestep,
            pred_original_sample,
            sample,
        )

        device = model_output.device
        noise = torch.randn(
            sample.size(),
            dtype=sample.dtype,
            layout=sample.layout,
            device=device,
            generator=_compatible_generator(generator, device),
        )
        variance = self._get_variance(
            timestep,
            previous_timestep,
            predicted_variance=predicted_variance,
            device=sample.device,
            dtype=sample.dtype,
        ).sqrt() * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample

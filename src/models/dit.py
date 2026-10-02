# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# References:
# DiT:  https://github.com/facebookresearch/DiT
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from src.models.attention import Attention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if train or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


#################################################################################
#                           4D Patch Embedding                                  #
#################################################################################

class PatchEmbed4D(nn.Module):
    """
    True 4D patchification via einops.

    Input:  (B, C, Z, X, Y, T)          (or (B, 2C, ...) when self_conditioning=True)
    Output: (B, N_patches, embed_dim)

    patch_size: (pz, px, py, pt)
    """

    def __init__(self, input_size, patch_size, in_channels, embed_dim, self_conditioning=False):
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.patch_size = tuple(int(v) for v in patch_size)  # (pz, px, py, pt)
        pz, px, py, pt = self.patch_size
        Z, X, Y, T = self.input_size

        for axis_name, dim, patch in zip("ZXYT", (Z, X, Y, T), self.patch_size):
            if dim % patch != 0:
                raise ValueError(
                    f"PatchEmbed4D: input_size axis {axis_name}={dim} not "
                    f"divisible by patch_size {patch}"
                )

        self.grid_size = [Z // pz, X // px, Y // py, T // pt]
        self.num_patches = int(np.prod(self.grid_size))

        # When self-conditioning is active the noisy input and the previous x0
        # estimate are concatenated channel-wise before patchification, so the
        # patch dimension doubles.
        patch_dim = in_channels * pz * px * py * pt
        if self_conditioning:
            patch_dim *= 2
        self.proj = nn.Linear(patch_dim, embed_dim)

    def grid_size_from_input(self, x_shape) -> tuple[int, int, int, int]:
        spatial_shape = tuple(int(v) for v in x_shape[-4:])
        grid = []
        for axis_name, dim, patch in zip("ZXYT", spatial_shape, self.patch_size):
            if dim % patch != 0:
                raise ValueError(
                    f"PatchEmbed4D: input axis {axis_name}={dim} is not "
                    f"divisible by patch_size {patch}"
                )
            grid.append(dim // patch)
        return tuple(grid)

    def forward(self, x):
        pz, px, py, pt = self.patch_size
        x = rearrange(
            x,
            'b c (z pz) (x px) (y py) (t pt) -> b (z x y t) (c pz px py pt)',
            pz=pz, px=px, py=py, pt=pt,
        )
        return self.proj(x)


#################################################################################
#                           Sine-Cosine Pos Embed 4D                            #
#################################################################################

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_4d_sincos_pos_embed(embed_dim, grid_size, spacing=None, patch_size=None):
    """
    Generate 4D sinusoidal positional embeddings.

    Parameters
    ----------
    embed_dim : int
        Must be divisible by 4.
    grid_size : (Z, X, Y, T)
        Number of patches per axis.
    spacing : sequence of 4 floats or None
        Physical size of one LATENT voxel along (Z, X, Y, T).  Code multiplies
        by ``patch_size`` internally to obtain per-patch physical extent, then
        normalises so the smallest axis step is 1.0 to keep position values in
        a sane range for sincos channels.  ``None`` reproduces the original
        isotropic (uniform integer grid) behaviour.

        For SA CMR with raw 1.7 mm in-plane / 10 mm slice spacing, stage1
        ds=8 spatial / ds=4 temporal, the latent voxel spacing is
        ``(10.0, 13.6, 13.6, 4.0)`` (Z keeps raw because stage1 is slice-wise).
    patch_size : sequence of 4 ints or None
        Required when ``spacing`` is set.  Patch size in latent space
        ``(pz, px, py, pt)`` — same value used by ``PatchEmbed4D``.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 4D pos embed"
    dim_each = embed_dim // 4

    if spacing is None:
        step = [1.0, 1.0, 1.0, 1.0]
    else:
        assert patch_size is not None, "patch_size must be provided when spacing is set"
        step = [float(s) * int(p) for s, p in zip(spacing, patch_size)]
        m = min(step)
        step = [s / m for s in step]

    z = np.arange(grid_size[0], dtype=np.float32) * step[0]
    x = np.arange(grid_size[1], dtype=np.float32) * step[1]
    y = np.arange(grid_size[2], dtype=np.float32) * step[2]
    t = np.arange(grid_size[3], dtype=np.float32) * step[3]

    grid = np.meshgrid(z, x, y, t, indexing='ij')
    grid = np.stack(grid, axis=0).reshape([4, -1])

    emb_z = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])
    emb_x = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])
    emb_y = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])
    emb_t = get_1d_sincos_pos_embed_from_grid(dim_each, grid[3])

    return np.concatenate([emb_z, emb_x, emb_y, emb_t], axis=1)


#################################################################################
#                           Core Transformer Blocks                             #
#################################################################################

class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        flash_attention=True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            use_flash_attention=flash_attention,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm if qk_norm else None,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), act_layer=nn.GELU, drop=mlp_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c, rope_positions=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, rope_positions=rope_positions)

        h2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h2)

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_volume, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


#################################################################################
#                              DiT for 4D Volumes                               #
#################################################################################

class DiT4D(nn.Module):
    """
    4D Diffusion Transformer operating on spatiotemporal latent volumes.

    Input:  (B, C, Z, X, Y, T)   — 4D latent volume (depth, height, width, time)
    Output: (B, C_out, Z, X, Y, T)

    patch_size: (pz, px, py, pt) — 4-tuple matching the paper's (1, 4, 4, 2).
    """

    def __init__(
        self,
        input_size,          # (Z, X, Y, T)
        patch_size,          # (pz, px, py, pt)
        in_channels,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        class_dropout_prob,
        num_classes,
        learn_sigma,
        flash_attention,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        self_conditioning: bool = False,
        temporal_pe: str = "absolute",
        pos_embed_mode: Optional[str] = None,
        max_input_size=None,
        variable_shape: bool = False,
        spacing=None,
        rope_temporal_scale: float = 1.0,
        qk_norm: bool = False,
        phase_schema: str = "legacy_v1",
    ):
        """
        Parameters
        ----------
        self_conditioning : bool
            If True the model accepts an optional ``x_self_cond`` (previous x0
            estimate) concatenated channel-wise with the noisy input before
            patchification.  Doubles the patch embedder's input dimension;
            output shape is unchanged.
        spacing : sequence of 4 floats or None
            Physical size of one LATENT voxel along (Z, X, Y, T).  Internally
            multiplied by ``patch_size`` and normalised so the smallest axis
            step = 1.0.  ``None`` gives isotropic (uniform integer grid)
            encoding.  See ``get_4d_sincos_pos_embed`` for details.
        temporal_pe : {"absolute", "interpolated", "phase", "rope", "time_rope", "rope4d"}
            Backwards-compatible positional mode selector. ``"rope"`` is the
            legacy temporal-only RoPE path and is treated as ``"time_rope"``.
            Prefer ``pos_embed_mode`` for new configs.
        pos_embed_mode : {"absolute", "interpolated", "interpolated_rescaled", "phase", "time_rope", "rope4d", "center_select"}
            Explicit CardioDiT-v2 positional mode. ``rope4d`` applies RoPE to
            Q/K using dynamic ``(z,x,y,t)`` coordinates and no additive PE.
        variable_shape : bool
            If True, forward accepts any patch-divisible shape up to
            ``max_input_size`` and unpatchifies with the current grid.
        """
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.max_input_size = tuple(int(v) for v in (max_input_size or self.input_size))
        self.variable_shape = bool(variable_shape)
        if not self.variable_shape:
            self.max_input_size = self.input_size
        if learn_sigma:
            raise ValueError(
                "learn_sigma=true is unsupported: CardioDiT has no learned-variance "
                "training loss or sampler path. Set model.params.learn_sigma=false."
            )
        self.learn_sigma = False
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.self_conditioning = self_conditioning
        self.spacing = tuple(float(v) for v in spacing) if spacing is not None else None
        self.rope_temporal_scale = float(rope_temporal_scale)
        self.phase_schema = str(phase_schema).lower()
        if self.phase_schema not in {"legacy_v1", "cyclic_v2"}:
            raise ValueError(
                "phase_schema must be 'legacy_v1' or 'cyclic_v2'; "
                f"got {phase_schema!r}"
            )

        raw_mode = str(pos_embed_mode if pos_embed_mode is not None else temporal_pe).lower()
        if raw_mode == "rope":
            raw_mode = "time_rope"
        valid_modes = {"absolute", "interpolated", "interpolated_rescaled", "phase", "time_rope", "rope4d", "center_select"}
        if raw_mode not in valid_modes:
            raise ValueError(
                "pos_embed_mode/temporal_pe must be one of absolute, interpolated, interpolated_rescaled, "
                f"phase, time_rope, rope4d, center_select; got {raw_mode!r}"
            )
        self.pos_embed_mode = raw_mode
        self.temporal_pe = raw_mode

        head_dim = hidden_size // num_heads
        if self.pos_embed_mode == "rope4d" and head_dim < 8:
            raise ValueError(
                "rope4d requires head_dim >= 8 so each z/x/y/t axis receives "
                f"at least one rotary pair; got hidden_size={hidden_size}, num_heads={num_heads}"
            )

        pz, px, py, pt = self.patch_size
        self.patch_volume = pz * px * py * pt

        self.x_embedder = PatchEmbed4D(
            self.max_input_size, self.patch_size, in_channels, hidden_size,
            self_conditioning=self_conditioning,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) if num_classes > 0 else None

        self.pos_grid_size = tuple(self.x_embedder.grid_size)
        if self.pos_embed_mode == "rope4d":
            self.register_parameter("pos_embed", None)
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.x_embedder.num_patches, hidden_size),
                requires_grad=False,
            )

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, mlp_ratio, flash_attention,
                attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop,
                qk_norm=qk_norm,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_volume, self.out_channels)

        Zg, Xg, Yg, Tg = self.pos_grid_size
        _, _, _, token_t = np.meshgrid(
            np.arange(Zg),
            np.arange(Xg),
            np.arange(Yg),
            np.arange(Tg),
            indexing="ij",
        )
        self.register_buffer(
            "token_time_index",
            torch.from_numpy(token_t.reshape(-1)).long(),
            persistent=False,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_init)

        if self.pos_embed is not None:
            pos_embed = get_4d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.pos_grid_size,
                spacing=self.spacing,
                patch_size=self.patch_size,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @staticmethod
    def _sincos_1d_torch(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        if embed_dim % 2 != 0:
            raise ValueError("1D sin-cos embedding dimension must be even")
        half = embed_dim // 2
        omega = torch.arange(half, dtype=torch.float32, device=pos.device)
        omega = 1.0 / (10000 ** (omega / half))
        out = pos.float().unsqueeze(-1) * omega
        return torch.cat([out.sin(), out.cos()], dim=-1)

    @staticmethod
    def _periodic_phase_embedding(embed_dim: int, phase: torch.Tensor) -> torch.Tensor:
        """Integer-harmonic embedding, exactly periodic under phase += 1."""
        if embed_dim % 2 != 0:
            raise ValueError("Periodic phase embedding dimension must be even")
        harmonics = torch.arange(
            1, embed_dim // 2 + 1, dtype=torch.float32, device=phase.device
        )
        canonical_phase = torch.remainder(phase.float(), 1.0)
        angles = canonical_phase.unsqueeze(-1) * harmonics * (2.0 * math.pi)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    @staticmethod
    def _circular_pool_phase(alpha: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pool frame phases into explicit patch bins without endpoint duplication."""
        source_length = int(alpha.shape[1])
        pooled = []
        for index in range(int(target_length)):
            start = (index * source_length) // int(target_length)
            end = ((index + 1) * source_length) // int(target_length)
            if end <= start:
                center = min(source_length - 1, int((index + 0.5) * source_length / target_length))
                values = alpha[:, center:center + 1]
            else:
                values = alpha[:, start:end]
            angles = values * (2.0 * math.pi)
            mean_angle = torch.atan2(angles.sin().mean(1), angles.cos().mean(1))
            pooled.append(torch.remainder(mean_angle / (2.0 * math.pi), 1.0))
        return torch.stack(pooled, dim=1)

    def _axis_steps(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.spacing is None:
            values = [1.0, 1.0, 1.0, 1.0]
        else:
            values = [
                float(s) * int(p)
                for s, p in zip(self.spacing, self.patch_size)
            ]
            scale = min(values)
            values = [v / scale for v in values]
        return torch.tensor(values, dtype=torch.float32, device=device)

    @staticmethod
    def _token_axis_indices(
        grid_size: tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        z, x, y, t = torch.meshgrid(
            torch.arange(grid_size[0], device=device),
            torch.arange(grid_size[1], device=device),
            torch.arange(grid_size[2], device=device),
            torch.arange(grid_size[3], device=device),
            indexing="ij",
        )
        return torch.stack((z, x, y, t), dim=-1).reshape(-1, 4)

    def _validate_input_grid(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        actual_size = tuple(int(v) for v in x.shape[-4:])
        if not self.variable_shape and actual_size != self.input_size:
            raise ValueError(
                f"DiT4D fixed-shape model expected input_size={self.input_size}, "
                f"got {actual_size}. Set variable_shape=true for dynamic grids."
            )
        if self.variable_shape:
            for axis_name, dim, max_dim in zip("ZXYT", actual_size, self.max_input_size):
                if dim > max_dim:
                    raise ValueError(
                        f"DiT4D variable-shape input axis {axis_name}={dim} exceeds "
                        f"max_input_size {self.max_input_size}"
                    )
        return self.x_embedder.grid_size_from_input(x.shape)

    def _center_select_pos_embed(
        self,
        grid_size: tuple[int, int, int, int],
    ) -> torch.Tensor:
        if tuple(grid_size) == self.pos_grid_size:
            return self.pos_embed

        if not self.variable_shape:
            raise ValueError(
                f"Positional grid {grid_size} differs from fixed grid {self.pos_grid_size}"
            )

        slices = []
        for actual, max_axis in zip(grid_size, self.pos_grid_size):
            if actual > max_axis:
                raise ValueError(
                    f"Requested PE grid {grid_size} exceeds max grid {self.pos_grid_size}"
                )
            start = (max_axis - actual) // 2
            slices.append(slice(start, start + actual))

        hidden = self.pos_embed.shape[-1]
        pos = self.pos_embed.reshape(1, *self.pos_grid_size, hidden)
        pos = pos[(slice(None), *slices, slice(None))]
        return pos.reshape(1, int(np.prod(grid_size)), hidden)

    def _base_positional_embedding(
        self,
        grid_size: tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        return self._center_select_pos_embed(grid_size).to(device)

    def _patch_time_positions(
        self,
        batch_size: int,
        device: torch.device,
        alpha_t: Optional[torch.Tensor],
        grid_size: Optional[tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        grid_size = tuple(grid_size or self.pos_grid_size)
        Tg = int(grid_size[3])
        step_t = self._axis_steps(device=device)[3] * self.rope_temporal_scale
        if alpha_t is None:
            token_t = self._token_axis_indices(grid_size, device)[:, 3]
            offset = 0.5 if self.phase_schema == "cyclic_v2" else 0.0
            return (
                torch.arange(Tg, device=device, dtype=torch.float32)[token_t] + offset
            ) * step_t

        alpha = torch.as_tensor(alpha_t, dtype=torch.float32).to(device=device)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(0).expand(batch_size, -1)
        elif alpha.ndim > 2:
            alpha = alpha.reshape(alpha.shape[0], -1)
        if alpha.shape[0] != batch_size:
            if alpha.shape[0] == 1:
                alpha = alpha.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"alpha_t batch size {alpha.shape[0]} does not match x batch {batch_size}"
                )

        alpha_min = float(alpha.detach().amin().cpu())
        alpha_max = float(alpha.detach().amax().cpu())
        is_unit_phase = alpha_min >= -1e-4 and alpha_max <= 1.0 + 1e-4
        if self.phase_schema == "cyclic_v2" and is_unit_phase:
            if alpha.shape[1] != Tg:
                alpha = self._circular_pool_phase(alpha, Tg)
            patch_positions = torch.remainder(alpha, 1.0) * Tg
        elif alpha.shape[1] != Tg:
            alpha = F.interpolate(
                alpha.unsqueeze(1),
                size=Tg,
                mode="linear",
                align_corners=True,
            ).squeeze(1)

            alpha_min = float(alpha.detach().amin().cpu())
            alpha_max = float(alpha.detach().amax().cpu())
            if alpha_min >= -1e-4 and alpha_max <= 1.0 + 1e-4:
                patch_positions = alpha * max(Tg - 1, 1)
            else:
                patch_positions = alpha
        elif is_unit_phase:
            patch_positions = alpha * max(Tg - 1, 1)
        else:
            patch_positions = alpha

        token_t = self._token_axis_indices(grid_size, device)[:, 3]
        return (patch_positions * step_t)[:, token_t]

    def _rope_coordinates_4d(
        self,
        batch_size: int,
        device: torch.device,
        alpha_t: Optional[torch.Tensor],
        grid_size: tuple[int, int, int, int],
    ) -> torch.Tensor:
        indices = self._token_axis_indices(grid_size, device).float()
        steps = self._axis_steps(device=device)
        spatial = indices[:, :3] * steps[:3]
        time_positions = self._patch_time_positions(
            batch_size, device, alpha_t, grid_size,
        )
        if time_positions.ndim == 1:
            coords = torch.cat((spatial, time_positions[:, None]), dim=-1)
            return coords.unsqueeze(0).expand(batch_size, -1, -1)
        spatial = spatial.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat((spatial, time_positions.unsqueeze(-1)), dim=-1)

    def _positional_embedding(
        self,
        batch_size: int,
        device: torch.device,
        alpha_t: Optional[torch.Tensor],
        grid_size: Optional[tuple[int, int, int, int]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        grid_size = tuple(grid_size or self.pos_grid_size)
        if self.pos_embed_mode == "rope4d":
            return None

        hidden = self.pos_embed.shape[-1]
        dtype = dtype or self.pos_embed.dtype
        base = self._base_positional_embedding(grid_size, device).to(dtype=dtype)
        if self.pos_embed_mode in {"absolute", "center_select"}:
            return base

        dim_each = hidden // 4
        positions = self._patch_time_positions(batch_size, device, alpha_t, grid_size)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1)

        if self.pos_embed_mode == "interpolated_rescaled":
            # Preserve the TempAlign experiment's training-time temporal range.
            default_t = self.input_size[3] // self.patch_size[3]
            positions_t = torch.linspace(
                0.0, max(default_t - 1, 0), grid_size[3], device=device,
            ) * self._axis_steps(device=device)[3]
            token_t = self._token_axis_indices(grid_size, device)[:, 3]
            positions = positions_t[token_t].unsqueeze(0).expand(batch_size, -1)

        if self.pos_embed_mode in {"interpolated", "interpolated_rescaled"}:
            pos = base.expand(batch_size, -1, -1).clone()
            pos[..., 3 * dim_each:4 * dim_each] = self._sincos_1d_torch(
                dim_each, positions
            )
            return pos

        if self.pos_embed_mode == "phase":
            pos = base.expand(batch_size, -1, -1)
            denom = self._axis_steps(device=device)[3] * max(grid_size[3] - 1, 1)
            phase = positions / denom.clamp_min(1.0)
            if self.phase_schema == "cyclic_v2":
                cyclic_denom = self._axis_steps(device=device)[3] * max(grid_size[3], 1)
                phase = torch.remainder(positions / cyclic_denom.clamp_min(1.0), 1.0)
                pos = pos.clone()
                pos[..., 3 * dim_each:4 * dim_each] = 0.0
                phase_embed = self._periodic_phase_embedding(hidden, phase)
            else:
                phase_embed = self._sincos_1d_torch(hidden, phase * (2.0 * math.pi))
            return pos + phase_embed.to(dtype=pos.dtype)

        if self.pos_embed_mode == "time_rope":
            pos = base.expand(batch_size, -1, -1).clone()
            pos[..., 3 * dim_each:4 * dim_each] = 0.0
            return pos

        raise AssertionError(f"Unhandled pos_embed_mode {self.pos_embed_mode}")

    def _rope_positions(
        self,
        batch_size: int,
        device: torch.device,
        alpha_t: Optional[torch.Tensor],
        grid_size: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[torch.Tensor]:
        grid_size = tuple(grid_size or self.pos_grid_size)
        if self.pos_embed_mode == "time_rope":
            return self._patch_time_positions(batch_size, device, alpha_t, grid_size)
        if self.pos_embed_mode == "rope4d":
            return self._rope_coordinates_4d(batch_size, device, alpha_t, grid_size)
        return None

    def unpatchify(self, x, grid_size: Optional[tuple[int, int, int, int]] = None):
        """
        (B, N, patch_volume * C_out)  ->  (B, C_out, Z, X, Y, T)
        """
        B = x.shape[0]
        pz, px, py, pt = self.patch_size
        C = self.out_channels
        Zg, Xg, Yg, Tg = tuple(grid_size or self.pos_grid_size)
        expected_tokens = Zg * Xg * Yg * Tg
        if x.shape[1] != expected_tokens:
            raise ValueError(
                f"Cannot unpatchify {x.shape[1]} tokens with grid {(Zg, Xg, Yg, Tg)} "
                f"({expected_tokens} tokens expected)"
            )

        x = x.reshape(B, Zg, Xg, Yg, Tg, pz, px, py, pt, C)
        x = x.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8)
        x = x.reshape(B, C, Zg * pz, Xg * px, Yg * py, Tg * pt)
        return x

    def forward(self, x, t, y=None, x_self_cond=None, alpha_t=None):
        """
        Parameters
        ----------
        x            : (B, C, Z, X, Y, T)   noisy 4D latent.
        t            : (B,)                  timestep indices.
        y            : (B,) or None          optional class labels.
        x_self_cond  : (B, C, Z, X, Y, T) or None
                       Previous x0 estimate for self-conditioning.
                       Only consumed when ``self_conditioning=True``; zeros are
                       substituted automatically when None is passed.
        alpha_t      : (B, T) or (B, T_patches) or None
                       Optional temporal phase/descriptor values.
        """
        B = x.shape[0]
        grid_size = self._validate_input_grid(x)
        if self.self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            elif tuple(x_self_cond.shape[-4:]) != tuple(x.shape[-4:]):
                raise ValueError(
                    f"x_self_cond shape {tuple(x_self_cond.shape[-4:])} does not "
                    f"match x shape {tuple(x.shape[-4:])}"
                )
            x = torch.cat([x, x_self_cond], dim=1)  # (B, 2C, Z, X, Y, T)

        x = self.x_embedder(x)
        pos_embed = self._positional_embedding(
            B, x.device, alpha_t, grid_size, dtype=x.dtype,
        )
        if pos_embed is not None:
            x = x + pos_embed
        t = self.t_embedder(t)                      # (B, D)

        if self.y_embedder is not None and y is not None:
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        rope_positions = self._rope_positions(B, x.device, alpha_t, grid_size)
        for block in self.blocks:
            x = block(x, c, rope_positions=rope_positions)

        x = self.final_layer(x, c)
        return self.unpatchify(x, grid_size)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp

from src.models.attention import Attention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
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


class PatchEmbed4D(nn.Module):
    """
    True 4D patchification.

    Input: (B, C, Z, H, W, T)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(self, input_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = tuple(patch_size)
        pz, ph, pw, pt = self.patch_size
        z, h, w, t = input_size

        if z % pz or h % ph or w % pw or t % pt:
            raise ValueError(
                "input_size must be divisible by patch_size along every axis"
            )

        self.grid_size = [z // pz, h // ph, w // pw, t // pt]
        self.num_patches = int(np.prod(self.grid_size))
        patch_dim = in_channels * pz * ph * pw * pt
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        pz, ph, pw, pt = self.patch_size
        x = rearrange(
            x,
            "b c (z pz) (h ph) (w pw) (t pt) -> b (z h w t) (c pz ph pw pt)",
            pz=pz,
            ph=ph,
            pw=pw,
            pt=pt,
        )
        return self.proj(x)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 1D sincos embeddings")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_4d_sincos_pos_embed(embed_dim, grid_size):
    if embed_dim % 8 != 0:
        raise ValueError("embed_dim must be divisible by 8 for 4D sincos embeddings")

    dim_each = embed_dim // 4
    z = np.arange(grid_size[0], dtype=np.float32)
    h = np.arange(grid_size[1], dtype=np.float32)
    w = np.arange(grid_size[2], dtype=np.float32)
    t = np.arange(grid_size[3], dtype=np.float32)

    grid = np.meshgrid(z, h, w, t, indexing="ij")
    grid = np.stack(grid, axis=0).reshape([4, -1])

    emb_z = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])
    emb_t = get_1d_sincos_pos_embed_from_grid(dim_each, grid[3])
    return np.concatenate([emb_z, emb_h, emb_w, emb_t], axis=1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        flash_attention=True,
        attn_drop=0.0,
        mlp_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            use_flash_attention=flash_attention,
            attn_drop=attn_drop,
            proj_drop=attn_drop,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            hidden_size,
            int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
            drop=mlp_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
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


class DiT4D(nn.Module):
    """
    Diffusion Transformer for latent 4D CMR volumes.

    Input and output shape: (B, C, Z, H, W, T).
    """

    def __init__(
        self,
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        class_dropout_prob,
        num_classes,
        learn_sigma,
        flash_attention,
        attn_drop=0.0,
        mlp_drop=0.0,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.patch_size = tuple(patch_size)
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        pz, ph, pw, pt = self.patch_size
        self.patch_volume = pz * ph * pw * pt

        self.x_embedder = PatchEmbed4D(
            self.input_size, self.patch_size, in_channels, hidden_size
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = (
            LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            if num_classes > 0
            else None
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    flash_attention,
                    attn_drop,
                    mlp_drop,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, self.patch_volume, self.out_channels
        )

        self.initialize_weights()

    def initialize_weights(self):
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(init_linear)

        pos_embed = get_4d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.grid_size
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

    def unpatchify(self, x):
        bsz = x.shape[0]
        pz, ph, pw, pt = self.patch_size
        channels = self.out_channels
        zg, hg, wg, tg = self.x_embedder.grid_size

        x = x.reshape(bsz, zg, hg, wg, tg, pz, ph, pw, pt, channels)
        x = x.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8)
        x = x.reshape(bsz, channels, zg * pz, hg * ph, wg * pw, tg * pt)
        return x

    def forward(self, x, t, y=None):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)

        if self.y_embedder is not None and y is not None:
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return self.unpatchify(x)

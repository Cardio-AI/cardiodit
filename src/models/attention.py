from __future__ import annotations

import importlib.util
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if importlib.util.find_spec("xformers") is not None:
    import xformers.ops as xops

    HAS_XFORMERS = True
else:
    xops = None
    HAS_XFORMERS = False


class Attention(nn.Module):
    """Multi-head self-attention with xFormers or PyTorch SDPA fallback."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash_attention = use_flash_attention and HAS_XFORMERS

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            bsz, num_tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)

        if self.use_flash_attention:
            x = xops.memory_efficient_attention(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                attn_bias=None,
            )
        elif hasattr(F, "scaled_dot_product_attention"):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2)
        else:
            query = query.transpose(1, 2) * self.scale
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            attn = query @ key.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = (attn @ value).transpose(1, 2)

        x = x.reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

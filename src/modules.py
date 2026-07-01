"""
Shared building blocks for HFM-1D: feed-forward, patch embedding, learned 2-D
positional bias, a shallow skip encoder, and pre-norm self-/cross-attention
transformer blocks.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import List

from .attention import CrossAttention


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """[B, C, H, W] → [B, P, P, embed_dim] via strided Conv2d."""

    def __init__(self, in_channels: int, patch_px: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_px, stride=patch_px, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), 'b c h w -> b h w c')


class LearnedPos2D(nn.Module):
    """Additive learned 2-D positional bias."""

    def __init__(self, h: int, w: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, h, w, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


# ---------------------------------------------------------------------------
# Shallow skip encoder (initial-frame detail anchor for the decoder)
# ---------------------------------------------------------------------------

class SkipEncoder(nn.Module):
    """Single-scale, high-resolution features from the raw input frame."""

    def __init__(self, in_channels: int, skip_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, skip_ch, 3, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.stem(x)]


# ---------------------------------------------------------------------------
# Transformer blocks (pre-norm)
# ---------------------------------------------------------------------------

class SelfAttnBlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention (queries ← context) + FFN."""

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn    = CrossAttention(q_dim, kv_dim, n_heads, dropout)
        self.norm2   = nn.LayerNorm(q_dim)
        self.ffn     = FeedForward(q_dim, mlp_ratio, dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv))
        q = q + self.ffn(self.norm2(q))
        return q

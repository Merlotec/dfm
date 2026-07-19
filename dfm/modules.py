"""
Shared building blocks for DFM: feed-forward, patch embedding, learned 2-D
positional bias, a shallow skip encoder, and pre-norm self-/cross-attention
transformer blocks.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Union, List, Optional

from .attention import CrossAttention, LocalSelfAttention2D
from .config import DFMConfig


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
def sincos_2d(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """
    Deterministic 2-D sinusoidal positional encoding → [grid_h*grid_w, dim].

    Half the channels encode the row coordinate, half the column, each with the
    standard sin/cos frequency ladder.  Being a fixed function of position (not a
    learned parameter), it is recomputed for any grid shape (H≠W allowed), so it
    carries no resolution-specific weights.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sin-cos"

    def _1d(pos: torch.Tensor, d: int) -> torch.Tensor:
        omega = 1.0 / (10000 ** (torch.arange(d // 2, dtype=torch.float) / (d // 2)))
        ang = pos[:, None].float() * omega[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=1)          # [N, d]

    rows = torch.arange(grid_h).repeat_interleave(grid_w)        # [grid_h·grid_w]
    cols = torch.arange(grid_w).repeat(grid_h)                    # [grid_h·grid_w]
    return torch.cat([_1d(rows, dim // 2), _1d(cols, dim // 2)], dim=1)  # [grid_h·grid_w, dim]
class SelfAttnBlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        n = self.norm1(x)
        # causal=True → token i attends only to 0..i (prefix mask over the ordered slot
        # axis), so the first-N tokens are invariant to the total count.  The explicit
        # mask is required by nn.MultiheadAttention; is_causal lets it take the fast path.
        attn_mask = None
        if causal:
            L = x.shape[1]
            attn_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=x.device, dtype=n.dtype), diagonal=1)
        x = x + self.attn(n, n, n, need_weights=False, attn_mask=attn_mask, is_causal=causal)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class LocalSelfAttnBlock(nn.Module):
    """Pre-norm local (2r+1)² windowed self-attention with 2-D RoPE + FFN.

    Operates on a flattened grid_h×grid_w patch grid: input/output [B, grid_h*grid_w, dim].
    """

    def __init__(self, dim: int, n_heads: int, grid_h: int, grid_w: int, radius: int = 1,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = LocalSelfAttention2D(dim, n_heads, grid_h, grid_w, radius, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
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

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                key_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), key_bias=key_bias)
        q = q + self.ffn(self.norm2(q))
        return q


# ---------------------------------------------------------------------------
# Ordered / nested slots (Matryoshka-style): per-example monotone weight ramp
# ---------------------------------------------------------------------------

def add_relative_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian noise scaled to each token's own RMS (per-slot, scale-invariant).

    Used as a denoising / rollout-stability regularizer on the latent slots: because
    the noise is proportional to each slot's magnitude, the near-null anchor and the
    low-energy late slots of an ordered latent are perturbed proportionally, not
    swamped.  `std` is a fraction of that RMS (e.g. 0.1 → 10 %).
    """
    if std <= 0:
        return x
    rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt()   # [..., 1]
    return x + std * rms.to(x.dtype) * torch.randn_like(x)
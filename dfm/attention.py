"""
Attention primitives for HFM-1D.

  - CrossAttention        : queries attend to a separate key/value set.
  - LocalSelfAttention2D  : each patch on a P×P grid attends only to its
                            (2r+1)² neighbourhood, with 2-D rotary position
                            embedding.  Neighbourhoods are gathered explicitly
                            (im2col-style) so cost is O(N·(2r+1)²), not O(N²).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v: [..., L, head_dim].  Returns [..., Lq, head_dim]."""
    scale = math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# 2-D rotary position embedding
# ---------------------------------------------------------------------------

def build_2d_rope(P: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [P*P, head_dim] for a P×P grid.

    The head dim is split in four: the first half encodes the row coordinate,
    the second half the column coordinate (each duplicated for the rotate-half
    trick), giving a genuine 2-D relative position bias under the dot product.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2-D RoPE"
    quarter = head_dim // 4
    theta   = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float) / quarter))
    rows    = torch.arange(P).repeat_interleave(P).float()   # [P²]
    cols    = torch.arange(P).repeat(P).float()              # [P²]
    rf      = torch.outer(rows, theta)                       # [P², quarter]
    cf      = torch.outer(cols, theta)
    half_c  = torch.cat([rf.cos(), cf.cos()], dim=-1)        # [P², hd/2]
    half_s  = torch.cat([rf.sin(), cf.sin()], dim=-1)
    cos     = torch.cat([half_c, half_c], dim=-1)            # [P², hd]
    sin     = torch.cat([half_s, half_s], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Local (windowed) self-attention on a P×P grid, with 2-D RoPE
# ---------------------------------------------------------------------------

class LocalSelfAttention2D(nn.Module):
    """
    Each of the P² patch tokens attends only to the (2r+1)² patches in its
    neighbourhood (r=1 → the token itself + its 8 neighbours).  Neighbourhoods
    are gathered by padding+shifting the grid, so the attention is over
    (2r+1)² keys per query instead of all P² — O(N·9) rather than O(N²).

    Input / output: [B, P*P, dim]  (row-major flattening of the grid).
    """

    def __init__(self, dim: int, n_heads: int, grid: int, radius: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.P        = grid
        self.radius   = radius
        self.win      = 2 * radius + 1

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop     = nn.Dropout(dropout)

        # RoPE tables for centre queries and for gathered neighbour keys
        cos, sin = build_2d_rope(grid, self.head_dim)        # [N, hd]
        cos_grid = cos.reshape(grid, grid, self.head_dim)
        sin_grid = sin.reshape(grid, grid, self.head_dim)
        # Deterministic, resolution-shaped → non-persistent (rebuilt at construction,
        # never stored in the checkpoint, so weights transfer across resolutions).
        self.register_buffer('cos_c', cos, persistent=False)                   # [N, hd]
        self.register_buffer('sin_c', sin, persistent=False)
        self.register_buffer('cos_n', self._gather_grid(cos_grid), persistent=False)  # [N, W, hd]
        self.register_buffer('sin_n', self._gather_grid(sin_grid), persistent=False)
        # validity mask: which of the W neighbours are inside the grid
        ones = torch.ones(grid, grid, 1)
        mask = self._gather_grid(ones).squeeze(-1) > 0.5     # [N, W]
        self.register_buffer('nbr_mask', mask, persistent=False)

    cos_c: torch.Tensor
    sin_c: torch.Tensor
    cos_n: torch.Tensor
    sin_n: torch.Tensor
    nbr_mask: torch.Tensor

    def _gather_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """[P, P, X] → [P*P, win*win, X] of neighbourhood values (zero-padded)."""
        P, r, win = self.P, self.radius, self.win
        X = grid.shape[-1]
        g = grid.permute(2, 0, 1).unsqueeze(0)               # [1, X, P, P]
        g = F.pad(g, (r, r, r, r))
        cols = [g[:, :, di:di + P, dj:dj + P]
                for di in range(win) for dj in range(win)]
        nb = torch.stack(cols, dim=2)                        # [1, X, W, P, P]
        return nb.permute(0, 3, 4, 2, 1).reshape(P * P, win * win, X)

    def _gather_batch(self, t: torch.Tensor) -> torch.Tensor:
        """[B, N, C] → [B, N, win*win, C] of neighbourhood features."""
        B, N, C = t.shape
        P, r, win = self.P, self.radius, self.win
        g = t.reshape(B, P, P, C).permute(0, 3, 1, 2)        # [B, C, P, P]
        g = F.pad(g, (r, r, r, r))
        cols = [g[:, :, di:di + P, dj:dj + P]
                for di in range(win) for dj in range(win)]
        nb = torch.stack(cols, dim=2)                        # [B, C, W, P, P]
        return nb.permute(0, 3, 4, 2, 1).reshape(B, N, win * win, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        nh, hd, W = self.n_heads, self.head_dim, self.win * self.win

        q  = self.q_proj(x).reshape(B, N, nh, hd)
        kn = self._gather_batch(self.k_proj(x)).reshape(B, N, W, nh, hd)
        vn = self._gather_batch(self.v_proj(x)).reshape(B, N, W, nh, hd)

        # 2-D RoPE: centre position on q, neighbour positions on k
        q  = apply_rope(q,  self.cos_c.view(1, N, 1, hd),    self.sin_c.view(1, N, 1, hd))
        kn = apply_rope(kn, self.cos_n.view(1, N, W, 1, hd), self.sin_n.view(1, N, W, 1, hd))

        q  = q.permute(0, 2, 1, 3)                            # [B, nh, N, hd]
        kn = kn.permute(0, 3, 1, 2, 4)                        # [B, nh, N, W, hd]
        vn = vn.permute(0, 3, 1, 2, 4)

        scores = (q.unsqueeze(3) * kn).sum(-1) / math.sqrt(hd)   # [B, nh, N, W]
        scores = scores.masked_fill(~self.nbr_mask.view(1, 1, N, W), float('-inf'))
        attn   = scores.softmax(dim=-1)

        out = (attn.unsqueeze(-1) * vn).sum(3)                # [B, nh, N, hd]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.drop(self.out_proj(out))


class CrossAttention(nn.Module):
    """
    Standard multi-head cross-attention.

    queries: [B, Lq, q_dim]
    context: [B, Lk, kv_dim]   (kv_dim may differ from q_dim)
    returns: [B, Lq, q_dim]
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert q_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = q_dim // n_heads
        self.q_proj   = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj   = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj   = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Lq, _ = queries.shape
        Lk = context.shape[1]
        nh, hd = self.n_heads, self.head_dim

        q = self.q_proj(queries).reshape(B, Lq, nh, hd).transpose(1, 2)
        k = self.k_proj(context).reshape(B, Lk, nh, hd).transpose(1, 2)
        v = self.v_proj(context).reshape(B, Lk, nh, hd).transpose(1, 2)

        out = _scaled_dot(q, k, v).transpose(1, 2).reshape(B, Lq, nh * hd)
        return self.drop(self.out_proj(out))

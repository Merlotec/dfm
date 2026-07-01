"""
Attention primitives for HFM-1D.

Only multi-head cross-attention is needed: the Perceiver-style encoder, the slot
decoder, and the context injection in the evolution operator all reduce to
queries from one token set attending to keys/values from another.
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

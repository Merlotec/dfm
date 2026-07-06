"""
ContextEncoder: summarises "which dynamics" into K context tokens [B, K, d_ctx].

Instead of re-encoding raw history pixels, it operates on the *frozen-AE latents*
of the context frames — each context frame X_j is encoded (anchored at the rollout
start X_0) as  L_j = AE.encode(X_0, X_j), and the encoder aggregates the set of
those latents into the conditioning tokens.  This reuses the AE's representation
(no second image encoder) and keeps the conditioning in the same latent space the
evolution operator works in.

Input:  ctx_latents [B, F, n_slots, d_model]  (F = n_context frames)
Output: context     [B, n_ctx_tokens, d_ctx]
"""

import torch
import torch.nn as nn

from .config import DFMConfig
from .modules import FeedForward, SelfAttnBlock
from .attention import CrossAttention


class ContextEncoder(nn.Module):
    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg

        # AE latents are d_model; the conditioning space is d_ctx
        self.in_proj = (nn.Linear(cfg.d_model, cfg.d_ctx)
                        if cfg.d_model != cfg.d_ctx else nn.Identity())
        self.frame_emb = nn.Embedding(64, cfg.d_ctx)   # which context frame (history position)

        self.layers = nn.ModuleList([
            SelfAttnBlock(cfg.d_ctx, cfg.n_ctx_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_ctx_layers)
        ])

        # K learned query tokens; Perceiver-style cross-attn produces the summary
        self.summary_tokens   = nn.Parameter(torch.zeros(1, cfg.n_ctx_tokens, cfg.d_ctx))
        self.summary_cross    = CrossAttention(cfg.d_ctx, cfg.d_ctx, cfg.n_ctx_heads, cfg.dropout)
        self.summary_norm_q   = nn.LayerNorm(cfg.d_ctx)
        self.summary_norm_kv  = nn.LayerNorm(cfg.d_ctx)
        self.summary_ffn      = FeedForward(cfg.d_ctx, cfg.mlp_ratio, cfg.dropout)
        self.summary_norm_ffn = nn.LayerNorm(cfg.d_ctx)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.summary_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ctx_latents: torch.Tensor) -> torch.Tensor:
        """ctx_latents: [B, F, n_slots, d_model] → context [B, n_ctx_tokens, d_ctx]."""
        B, F, K, _ = ctx_latents.shape
        x = self.in_proj(ctx_latents)                             # [B, F, K, d_ctx]
        x = x + self.frame_emb.weight[:F].view(1, F, 1, -1)       # tag which frame
        x = x.reshape(B, F * K, -1)                               # [B, F·K, d_ctx]
        for layer in self.layers:
            x = layer(x)

        q  = self.summary_tokens.expand(B, -1, -1)
        kv = self.summary_norm_kv(x)
        q  = q + self.summary_cross(self.summary_norm_q(q), kv)
        q  = q + self.summary_ffn(self.summary_norm_ffn(q))
        return q                                                  # [B, n_ctx_tokens, d_ctx]

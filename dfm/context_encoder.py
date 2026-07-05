"""
ContextEncoder: learns the underlying PDE dynamics from a sequence of history
frames and summarises them into K context tokens [B, K, d_ctx].

These tokens condition the latent evolution operator (via cross-attention at
every rollout step), telling it *which* dynamics to integrate.  The summary is
produced by learned query vectors (Perceiver-style) that cross-attend to all
T·P² spatiotemporal frame tokens.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Optional

from .config import DFMConfig
from .modules import PatchEmbed, FeedForward, SelfAttnBlock, sincos_2d
from .attention import CrossAttention


class ContextEncoder(nn.Module):
    spatial_pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        P = cfg.img_size // cfg.ctx_patch_px

        self.patch_embed  = PatchEmbed(cfg.in_channels + 1, cfg.ctx_patch_px, cfg.d_ctx)
        self.register_buffer('spatial_pos', sincos_2d(P, cfg.d_ctx).unsqueeze(0),
                             persistent=False)             # [1, P², d_ctx]
        self.temporal_pos = nn.Embedding(64, cfg.d_ctx)   # up to 64 input frames

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

        self.cfg = cfg
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

    def forward(self, frames: List[torch.Tensor],
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = frames[0].shape[0]
        H, W = frames[0].shape[2], frames[0].shape[3]

        if pixel_mask is not None:
            mask_ch = pixel_mask.float().expand(B, 1, H, W)
        else:
            mask_ch = torch.ones(B, 1, H, W, device=frames[0].device, dtype=frames[0].dtype)

        tokens: List[torch.Tensor] = []
        for t, frame in enumerate(frames):
            if pixel_mask is not None:
                frame = frame * pixel_mask
            frame_aug = torch.cat([frame, mask_ch], dim=1)
            tok = rearrange(self.patch_embed(frame_aug), 'b h w d -> b (h w) d')  # [B, P², d_ctx]
            tok = tok + self.spatial_pos + self.temporal_pos.weight[t]
            tokens.append(tok)

        x = torch.cat(tokens, dim=1)                             # [B, T·P², d_ctx]
        for layer in self.layers:
            x = layer(x)

        q  = self.summary_tokens.expand(B, -1, -1)
        kv = self.summary_norm_kv(x)
        q  = q + self.summary_cross(self.summary_norm_q(q), kv)
        q  = q + self.summary_ffn(self.summary_norm_ffn(q))
        return q                                                 # [B, K, d_ctx]

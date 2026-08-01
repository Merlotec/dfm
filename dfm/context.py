"""Context encoder: summarise a run's governing regime from a block of frames.

The transport latent encode(X_{s-1}, X_s) describes HOW the fluid moved between two
frames, but not the regime it moved in -- viscosity, Reynolds number, inflow speed,
boundary conditions.  Two runs with different viscosity can look nearly identical
over one frame pair and diverge completely over a rollout, so evo is being asked to
extrapolate a trajectory whose governing parameters it cannot see.

This reads `n_context_frames` frames sampled from a RANDOM offset in the same run
(FVMSequenceDataset.random_context) and distils them into a small set of summary
tokens.  The decoupling is the point: because the context block is not adjacent to
the prediction window, the only thing it can usefully carry is the run's persistent
regime rather than a snapshot of the state just before the seed.

Those frames were already being loaded and thrown away -- every consumer unpacked
`for _, pred_b in ...` -- so this uses data the loader was already paying to render.

Lives in PHASE 2 (dynamics), conditioning evo and the closure heads.  The phase-1
autoencoder is untouched, so an existing AE checkpoint stays valid.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .config import DFMConfig
from .modules import PatchEmbed, SelfAttnBlock, CrossAttnBlock, sincos_2d


class ContextEncoder(nn.Module):
    """[B, T, C, H, W] context frames -> [B, n_ctx_tokens, d_model] summary."""

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        P = cfg.img_size // cfg.ctx_patch_px
        self.n_patch = P * P

        # mask channel goes in with the frame: geometry is part of the regime
        self.patch_embed = PatchEmbed(cfg.in_channels + cfg.n_mask_ch, cfg.ctx_patch_px, d)
        self.register_buffer('pos', sincos_2d(P, P, d).unsqueeze(0), persistent=False)
        # frames are ORDERED but their absolute offset in the run is arbitrary, so this
        # is a within-block index only
        self.time_emb = nn.Embedding(max(1, cfg.n_context_frames), d)

        self.layers = nn.ModuleList([
            SelfAttnBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_ctx_layers)
        ])
        # Perceiver-style distillation: a few learned queries pool the whole block,
        # so the output size is independent of T and of the patch grid.
        self.summary = nn.Parameter(torch.zeros(1, cfg.n_ctx_tokens, d))
        self.cross = CrossAttnBlock(d, d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.norm = nn.LayerNorm(d)

        nn.init.trunc_normal_(self.summary, std=0.02)
        nn.init.trunc_normal_(self.time_emb.weight, std=0.02)

    def forward(self, ctx: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C, H, W = ctx.shape
        nmc = self.cfg.n_mask_ch
        if pixel_mask is not None:
            valid = pixel_mask[:, :1].float()
            m = pixel_mask[:, :nmc].float()
            if valid.shape[0] != B:
                valid, m = valid.expand(B, -1, -1, -1), m.expand(B, -1, -1, -1)
        else:
            valid = None
            m = torch.ones(B, nmc, H, W, device=ctx.device, dtype=ctx.dtype)

        toks = []
        for t in range(T):
            f = ctx[:, t]
            if valid is not None:
                f = f * valid
            x = torch.cat([f, m.to(f.dtype)], dim=1)
            tok = rearrange(self.patch_embed(x), 'b h w d -> b (h w) d') + self.pos
            toks.append(tok + self.time_emb.weight[t])
        x = torch.cat(toks, dim=1)                       # [B, T*P^2, d]
        for blk in self.layers:
            x = blk(x)
        q = self.cross(self.summary.expand(B, -1, -1), x)
        return self.norm(q)                              # [B, n_ctx_tokens, d]

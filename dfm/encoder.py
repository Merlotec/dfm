"""
FrameEncoder: a single frame → a compact set of slot tokens.

This is the single-frame analogue of the context encoder.  The frame (plus its
geometry-mask channel) is patch-embedded, given a learned 2-D positional bias,
contextualised by a stack of self-attention blocks, and finally distilled into
`n_slots` learned query tokens via Perceiver-style cross-attention.

Only the slots are returned — the contextualised patch tokens are intentionally
discarded so that the entire rollout is forced through the slot bottleneck.
Fine spatial detail reaches the decoder via a separate shallow skip encoder on
the raw frame, not through these features.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .config import HFM1DConfig
from .modules import PatchEmbed, LearnedPos2D, SelfAttnBlock, CrossAttnBlock


class FrameEncoder(nn.Module):
    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.n_patch

        # +1 input channel for the geometry mask
        self.patch_embed = PatchEmbed(cfg.in_channels + 1, cfg.patch_px, cfg.d_model)
        self.spatial_pos = LearnedPos2D(P, P, cfg.d_model)

        self.layers = nn.ModuleList([
            SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_enc_layers)
        ])

        # Perceiver distillation into slot tokens
        self.slots       = nn.Parameter(torch.zeros(1, cfg.n_slots, cfg.d_model))
        self.slot_cross  = CrossAttnBlock(
            cfg.d_model, cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout
        )
        nn.init.trunc_normal_(self.slots, std=0.02)

    def forward(self, x_aug: torch.Tensor) -> torch.Tensor:
        """x_aug: [B, in_channels + 1, H, W]  →  slots [B, n_slots, d_model]."""
        B = x_aug.shape[0]

        tok = self.spatial_pos(self.patch_embed(x_aug))     # [B, P, P, d]
        tok = rearrange(tok, 'b h w d -> b (h w) d')        # [B, P², d]

        for layer in self.layers:
            tok = layer(tok)

        slots = self.slots.expand(B, -1, -1)
        slots = self.slot_cross(slots, tok)                 # [B, n_slots, d]
        return slots

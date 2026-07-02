"""
SlotDecoder: evolved slot tokens → image frame.

Two stages:

  1. Slot expansion.  A grid of learned patch-position queries cross-attends to
     the slots, re-materialising a P×P grid of patch tokens from the compact
     latent.  This is the inverse of the encoder's Perceiver distillation.

  2. Overlapping-patch synthesis.  Each patch token predicts a 3p/2 × 3p/2 pixel
     region; the overlapping predictions are blended with a bilinear tent kernel
     (tent-weighted F.fold, normalised by a pre-computed denominator).  A
     residual post-conv fuses in the shallow skip features (the initial-frame
     detail anchor) to recover sub-patch structure and remove block artefacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import HFM1DConfig
from .modules import LearnedPos2D, CrossAttnBlock


class OverlappingPatchDecoder(nn.Module):
    """[B, P, P, d] patch tokens → [B, C, H, W] via tent-blended overlapping patches."""

    def __init__(self, d: int, out_channels: int, img_size: int, patch_size: int,
                 skip_ch: int = 0):
        super().__init__()
        assert patch_size % 4 == 0, "patch_size must be divisible by 4 for 25% overlap"
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.skip_ch      = skip_ch
        kernel            = patch_size + patch_size // 2   # 3p/2
        self.kernel       = kernel
        n                 = img_size // patch_size

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, out_channels * kernel * kernel),
        )

        mid = max(out_channels * 8, 64)
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_ch, mid, 3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 3, padding=1, padding_mode='replicate'),
        )

        coords = torch.arange(kernel).float() + 0.5
        centre = kernel / 2.0
        w1d    = (centre - (coords - centre).abs()).clamp(min=0)
        wk_2d  = w1d.unsqueeze(1) * w1d.unsqueeze(0)
        self.register_buffer('weight_kernel', wk_2d)

        wk_flat  = wk_2d.flatten()
        norm_in  = wk_flat.unsqueeze(0).unsqueeze(-1).expand(1, -1, n * n)
        norm_map = F.fold(
            norm_in.contiguous(),
            output_size=(img_size, img_size),
            kernel_size=kernel,
            stride=patch_size,
            padding=patch_size // 4,
        )
        self.register_buffer('norm_map', norm_map)

    weight_kernel: torch.Tensor
    norm_map: torch.Tensor

    def forward(self, patches: torch.Tensor,
                skip_feats: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        B, ph, pw, d = patches.shape
        P      = ph * pw
        C      = self.out_channels
        kernel = self.kernel
        p      = self.patch_size

        preds   = self.head(patches.reshape(B, P, d))
        wk_flat = self.weight_kernel.flatten()
        preds   = preds.reshape(B, P, C, kernel * kernel) * wk_flat
        preds   = preds.permute(0, 2, 3, 1).reshape(B, C * kernel * kernel, P)

        output = F.fold(
            preds.contiguous(),
            output_size=(self.img_size, self.img_size),
            kernel_size=kernel,
            stride=p,
            padding=p // 4,
        )
        output = output / self.norm_map.clamp(min=1e-6)

        post_in = (
            torch.cat([output, skip_feats[0]], dim=1)
            if self.skip_ch > 0 and skip_feats
            else output
        )
        return output + self.post_conv(post_in)


class SlotDecoder(nn.Module):
    """Slots → patch grid (cross-attention) → image (overlapping-patch fold)."""

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.n_patch

        # Learned patch-position queries that read the slots
        self.patch_queries = LearnedPos2D(P, P, cfg.d_model)
        self.query_base    = nn.Parameter(torch.zeros(1, P * P, cfg.d_model))
        self.slot_read     = CrossAttnBlock(
            cfg.d_model, cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout
        )
        nn.init.trunc_normal_(self.query_base, std=0.02)

        self.synth = OverlappingPatchDecoder(
            cfg.d_model, cfg.in_channels, cfg.img_size, cfg.patch_px, cfg.skip_ch
        )

    def forward(self, slots: torch.Tensor,
                skip_feats: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        B = slots.shape[0]
        P = self.cfg.n_patch

        q = self.query_base.expand(B, -1, -1)               # [B, P², d]
        q = q + self.patch_queries.pos.reshape(1, P * P, -1)
        patch_tokens = self.slot_read(q, slots)             # [B, P², d]
        patch_tokens = patch_tokens.reshape(B, P, P, self.cfg.d_model)

        return self.synth(patch_tokens, skip_feats)

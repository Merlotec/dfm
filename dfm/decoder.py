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
from .modules import CrossAttnBlock, LocalSelfAttnBlock, sincos_2d


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
        # deterministic + resolution-shaped → non-persistent
        self.register_buffer('weight_kernel', wk_2d, persistent=False)

        wk_flat  = wk_2d.flatten()
        norm_in  = wk_flat.unsqueeze(0).unsqueeze(-1).expand(1, -1, n * n)
        norm_map = F.fold(
            norm_in.contiguous(),
            output_size=(img_size, img_size),
            kernel_size=kernel,
            stride=patch_size,
            padding=patch_size // 4,
        )
        self.register_buffer('norm_map', norm_map, persistent=False)

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


class _DecoderBlock(nn.Module):
    """One decoder layer: read the slots (cross-attn), then mix patches (self-attn)."""

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cross = CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads,
                                    cfg.mlp_ratio, cfg.dropout)
        self.self_attn = LocalSelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.n_patch,
                                            cfg.local_attn_radius,
                                            cfg.mlp_ratio, cfg.dropout)

    def forward(self, q: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        q = self.cross(q, slots)      # patch queries read the slot latent
        q = self.self_attn(q)         # patch tokens mix spatially
        return q


class SlotDecoder(nn.Module):
    """Slots → patch grid (cross- + self-attention layers) → image (overlapping-patch fold)."""

    pos: torch.Tensor

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.n_patch

        # A single shared learned query, made position-specific by a
        # resolution-agnostic sin-cos encoding (non-persistent buffer).
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.register_buffer('pos', sincos_2d(P, cfg.d_model).unsqueeze(0),
                             persistent=False)                   # [1, P², d]
        self.layers = nn.ModuleList([
            _DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)
        ])
        nn.init.trunc_normal_(self.query, std=0.02)

        self.synth = OverlappingPatchDecoder(
            cfg.d_model, cfg.in_channels, cfg.img_size, cfg.patch_px, cfg.skip_ch
        )

    def forward(self, slots: torch.Tensor,
                skip_feats: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        B = slots.shape[0]
        P = self.cfg.n_patch

        q = self.query.expand(B, P * P, -1) + self.pos      # [B, P², d]
        for layer in self.layers:
            q = layer(q, slots)                             # cross-attn → self-attn
        patch_tokens = q.reshape(B, P, P, self.cfg.d_model)

        return self.synth(patch_tokens, skip_feats)

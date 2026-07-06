"""
SlotDecoder: latent slot tokens → image frame, via a full transformer decoder.

  1. Transformer decode.  Learned patch-position queries pass through
     `n_dec_layers` blocks of [cross-attention to the slots → global self-attention
     over the patch grid].  The self-attention lets patches coordinate globally
     (which is what keeps seams/artefacts under control), and there is no reason
     to restrict its capacity: in the two-phase design the decoder is trained as a
     pure autoencoder (no jointly-trained dynamics to "steal"), and frozen at
     rollout time — so a powerful decoder simply means better reconstruction.

  2. Overlapping-patch synthesis.  Tent-weighted F.fold blends the overlapping
     patch predictions; a residual post-conv fuses the shallow skip features
     (initial-frame detail anchor) for sub-patch structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import DFMConfig
from .modules import CrossAttnBlock, SelfAttnBlock, LocalSelfAttnBlock, sincos_2d


class OverlappingPatchDecoder(nn.Module):
    """[B, P, P, d] patch tokens → [B, C, H, W] via tent-blended overlapping patches."""

    def __init__(self, d: int, out_channels: int, img_h: int, img_w: int, patch_size: int,
                 skip_ch: int = 0):
        super().__init__()
        assert patch_size % 4 == 0, "patch_size must be divisible by 4 for 25% overlap"
        self.img_h        = img_h
        self.img_w        = img_w
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.skip_ch      = skip_ch
        kernel            = patch_size + patch_size // 2   # 3p/2 (patches stay square)
        self.kernel       = kernel
        n_h               = img_h // patch_size
        n_w               = img_w // patch_size

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, out_channels * kernel * kernel),
        )

        # FiLM skip fusion: the latent patch tokens produce per-location (γ, β) that
        # gate/scale the skip detail, fused by a 1×1 projection.  No free 3×3 conv on the
        # image → it can only reshape the existing skip detail, not hallucinate local texture.
        if skip_ch > 0:
            self.film_gen  = nn.Linear(d, 2 * skip_ch)
            # (γ,β) live at patch resolution → nearest-upsample (repeat) then a depthwise 3×3
            # to smooth the patch seams.  This "resize-conv" avoids F.interpolate, which
            # blows up CUDA Inductor's symbolic-shape simplifier during backward codegen.
            self.skip_up   = nn.Conv2d(2 * skip_ch, 2 * skip_ch, 3, padding=1,
                                       groups=2 * skip_ch, padding_mode='replicate')
            self.skip_proj = nn.Conv2d(skip_ch, out_channels, 1)

        coords = torch.arange(kernel).float() + 0.5
        centre = kernel / 2.0
        w1d    = (centre - (coords - centre).abs()).clamp(min=0)
        wk_2d  = w1d.unsqueeze(1) * w1d.unsqueeze(0)
        # deterministic + resolution-shaped → non-persistent
        self.register_buffer('weight_kernel', wk_2d, persistent=False)

        wk_flat  = wk_2d.flatten()
        norm_in  = wk_flat.unsqueeze(0).unsqueeze(-1).expand(1, -1, n_h * n_w)
        norm_map = F.fold(
            norm_in.contiguous(),
            output_size=(img_h, img_w),
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
            output_size=(self.img_h, self.img_w),
            kernel_size=kernel,
            stride=p,
            padding=p // 4,
        )
        output = output / self.norm_map.clamp(min=1e-6)

        if self.skip_ch > 0 and skip_feats:
            skip = skip_feats[0]                                      # [B, skip_ch, H, W]
            gb = self.film_gen(patches.reshape(B, P, d))             # [B, P, 2·skip_ch]
            gb = gb.reshape(B, ph, pw, -1).permute(0, 3, 1, 2)       # [B, 2·skip_ch, ph, pw]
            gb = gb.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)   # nearest → H×W
            gb = self.skip_up(gb)                                    # smooth the patch seams
            gamma, beta = gb[:, :self.skip_ch], gb[:, self.skip_ch:]
            skip_mod = (1.0 + gamma) * skip + beta                   # γ = 1 + γ_raw (identity init)
            output = output + self.skip_proj(skip_mod)
        return output


class _DecoderBlock(nn.Module):
    """Decoder layer: cross-attend the slots, then local (windowed) self-attention with 2-D
    RoPE over the patch grid — matching the encoder, so positions are relative (extrapolate
    across resolutions) and the self-attention is O(N·window) instead of O(N²).  Global
    coordination still arrives through the slots (every patch reads the same slot set)."""

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cross     = CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads,
                                        cfg.ae_mlp, cfg.dropout)
        self.self_attn = LocalSelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.n_patch_h,
                                            cfg.n_patch_w, cfg.local_attn_radius,
                                            cfg.ae_mlp, cfg.dropout)

    def forward(self, q: torch.Tensor, slots: torch.Tensor,
                key_bias: torch.Tensor | None = None) -> torch.Tensor:
        q = self.cross(q, slots, key_bias=key_bias)   # read the (masked) slot latent
        q = self.self_attn(q)                          # coordinate patches locally (RoPE)
        return q


class SlotDecoder(nn.Module):
    """Slots → patch grid (cross + self-attention transformer) → image (overlapping-patch fold)."""

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        Ph, Pw = cfg.n_patch_h, cfg.n_patch_w

        # A single shared learned query, made position-specific by a
        # resolution-agnostic sin-cos encoding (non-persistent buffer).
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.register_buffer('pos', sincos_2d(Ph, Pw, cfg.d_model).unsqueeze(0),
                             persistent=False)                   # [1, Ph·Pw, d]

        # Learned per-slot rank embedding: gives each slot a stable identity so the
        # decoder knows *which* rank it is reading/attenuating (ordered-slot anchor).
        self.rank_embed = nn.Parameter(torch.zeros(1, cfg.n_slots, cfg.d_model))

        # slot self-attention (causal over the priority axis → prefix-invariant), applied
        # to the latent before the patch queries read it.
        self.slot_layers = nn.ModuleList([
            SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.n_slot_layers)
        ])
        self.slot_causal = cfg.slot_hierarchy

        self.layers = nn.ModuleList([
            _DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)
        ])
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.rank_embed, std=0.02)

        img_h, img_w = cfg.img_hw
        self.synth = OverlappingPatchDecoder(
            cfg.d_model, cfg.in_channels, img_h, img_w, cfg.patch_px, cfg.skip_ch
        )

    def forward(self, slots: torch.Tensor,
                skip_feats: Optional[List[torch.Tensor]] = None,
                key_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = slots.shape[0]
        Ph, Pw = self.cfg.n_patch_h, self.cfg.n_patch_w

        slots = slots + self.rank_embed[:, :slots.shape[1]]  # rank tag (sliced if truncated)
        for blk in self.slot_layers:                         # causal slot mixing
            slots = blk(slots, causal=self.slot_causal)
        q = self.query.expand(B, Ph * Pw, -1) + self.pos    # [B, Ph·Pw, d]
        for layer in self.layers:
            q = layer(q, slots, key_bias)                   # cross-attn (masked) → self-attn
        patch_tokens = q.reshape(B, Ph, Pw, self.cfg.d_model)

        return self.synth(patch_tokens, skip_feats)

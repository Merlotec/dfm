"""
HFM-1D — latent-rollout fluid model.

Pipeline (single training example):

    x0 ──shallow skip encoder──────────────► skip_feats (initial-frame anchor)
    x0 ──frame encoder──► slots S0
                            │
              ┌─────────────┴──────────── for i = 0 .. horizon-1 ───────────┐
              │  Sᵢ₊₁ = Evolve(Sᵢ, context, i)      (weight-shared operator) │
              │  x̂ᵢ₊₁ = Decode(Sᵢ₊₁, skip_feats)                            │
              │  (every `reencode_every` steps: S ← Encode(x̂), refresh skip) │
              └──────────────────────────────────────────────────────────────┘

The evolution stream is a pure slot bottleneck; spatial detail is supplied only
by the skip anchor, which is refreshed on re-encode to limit drift.  `context`
is produced once by the ContextEncoder from the history frames and injected at
every step.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .config import HFM1DConfig
from .modules import SkipEncoder
from .encoder import FrameEncoder
from .evolution import EvolutionOperator
from .decoder import SlotDecoder


class HFM1D(nn.Module):
    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        self.skip_encoder = SkipEncoder(cfg.in_channels, cfg.skip_ch)
        self.encoder      = FrameEncoder(cfg)
        self.evolution    = EvolutionOperator(cfg)
        self.decoder      = SlotDecoder(cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # (1) Tendency head → zero, so every evolution step starts as the
        #     identity map and the dynamics are learned as a residual.
        nn.init.zeros_(self.evolution.tendency.head.weight)
        nn.init.zeros_(self.evolution.tendency.head.bias)

        # (2) Final synthesis layers → zero, so the decoder starts as a
        #     pass-through of the tent-folded output (refinement learned).
        for seq in [self.decoder.synth.head, self.decoder.synth.post_conv]:
            for child in reversed(list(seq.children())):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    break

    def _augment(self, x: torch.Tensor,
                 pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = x.shape
        if pixel_mask is not None:
            x = x * pixel_mask
            mask_ch = pixel_mask.float().expand(B, 1, H, W)
        else:
            mask_ch = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        return torch.cat([x, mask_ch], dim=1)

    def encode(self, x: torch.Tensor,
               pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(self._augment(x, pixel_mask))

    def forward(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        horizon: Optional[int] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """x0: [B, C, H, W], context: [B, K, d_ctx] → list of `horizon` frames."""
        horizon = horizon or self.cfg.horizon
        m = self.cfg.reencode_every

        if self.training and self.cfg.noise_std > 0.0:
            x0 = x0 + torch.randn_like(x0) * self.cfg.noise_std

        skip_feats = self.skip_encoder(x0 * pixel_mask if pixel_mask is not None else x0)
        slots = self.encode(x0, pixel_mask)

        preds: List[torch.Tensor] = []
        for i in range(horizon):
            slots = self.evolution(slots, context, i)
            pred  = self.decoder(slots, skip_feats)
            preds.append(pred)

            if m > 0 and (i + 1) % m == 0 and i + 1 < horizon:
                # Re-anchor: refresh slots and skip detail from the prediction.
                anchor = pred * pixel_mask if pixel_mask is not None else pred
                skip_feats = self.skip_encoder(anchor)
                slots = self.encode(pred, pixel_mask)

        return preds

"""Reconstruction loss for fluid frames."""

import torch
import torch.nn as nn
from typing import Optional


class FluidLoss(nn.Module):
    """MSE + L1 reconstruction loss over valid (non-hole) pixels only."""

    def __init__(self, l1_weight: float = 0.1,
                 pixel_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.l1_weight = l1_weight
        if pixel_mask is not None:
            self.register_buffer('pixel_mask', pixel_mask)
        else:
            self.pixel_mask: Optional[torch.Tensor] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`pixel_mask` overrides the constructor buffer -- pass the PER-SAMPLE mask
        [B, 1|2, H, W] when a batch mixes geometries, otherwise the shared one is
        used and every sample would be scored against the wrong collider."""
        m = pixel_mask if pixel_mask is not None else self.pixel_mask
        if m is None:
            d = pred - target
            return d.pow(2).mean() + self.l1_weight * d.abs().mean()
        # first channel is is_valid (fluid); supervise those pixels only.  Boolean
        # indexing would flatten across samples; weight instead so a per-sample mask
        # broadcasts and the normaliser counts only that sample's fluid pixels.
        w = m[:, :1].to(pred.dtype)
        if w.shape[0] != pred.shape[0]:
            w = w.expand(pred.shape[0], -1, -1, -1)
        d = (pred - target)
        n = w.expand_as(d).sum().clamp_min(1.0)
        return ((d.pow(2) * w).sum() / n) + self.l1_weight * ((d.abs() * w).sum() / n)

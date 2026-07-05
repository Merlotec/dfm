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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pixel_mask is None:
            d = pred - target
        else:
            mask = self.pixel_mask.expand_as(pred).bool()
            d = pred[mask] - target[mask]
        return d.pow(2).mean() + self.l1_weight * d.abs().mean()

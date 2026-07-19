from typing import Optional, Union
"""
Latent evolution operator (phase 2).

A single, weight-shared transformer advances the increment latents by one step:
slot self-attention over BOTH token streams — detail→transport attention is the
backscatter coupling (unresolved chaos bending tomorrow's resolved flow) — with
a learned step embedding for rollout position.

The operator defines a tendency  f(S, C) = dS/dτ  and integrates it with either
forward-Euler or a two-stage midpoint (RK2) rule.  Recomputing the tendency at
each RK stage (rather than freezing it) makes this a genuine learned integrator
rather than a frozen linearisation — accurate over the short horizons the
bottleneck is trained on.

The tendency head is zero-initialised, so at the start of training every step is
the identity map and the decoder repeatedly sees the encoder's slots; the
dynamics are then learned as a residual.
"""

import torch
import torch.nn as nn

from .config import DFMConfig
from .modules import SelfAttnBlock


class _Tendency(nn.Module):
    """Computes dS = f(S): slot self-mixing across both token streams."""

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_evo_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.d_model)
        # Zero-init → identity integrator at start of training
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        x = slots
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


class EvolutionOperator(nn.Module):
    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        self.tendency  = _Tendency(cfg)
        self.step_emb  = nn.Embedding(cfg.max_rollout, cfg.d_model)
        nn.init.trunc_normal_(self.step_emb.weight, std=0.02)
        # Learnable latent step size, initialised to dt = 1.
        self.log_dt = nn.Parameter(torch.zeros(()))

    def _f(self, slots: torch.Tensor, step_idx) -> torch.Tensor:
        cap = self.cfg.max_rollout - 1
        if torch.is_tensor(step_idx):
            # per-example step indices [B] → per-example embedding [B, 1, d]
            emb = self.step_emb(step_idx.clamp(max=cap)).unsqueeze(1)
        else:
            emb = self.step_emb(torch.tensor(min(step_idx, cap), device=slots.device))
        return self.tendency(slots + emb)

    def forward(self, slots: torch.Tensor, step_idx) -> torch.Tensor:
        """Advance slots by one latent step.  slots: [B, n_slots, d_model].
        `step_idx`: Python int or [B] tensor of per-example indices."""
        dt = torch.exp(self.log_dt)

        if self.cfg.integrator == 'euler':
            return slots + dt * self._f(slots, step_idx)

        if self.cfg.integrator == 'rk2':          # midpoint
            k1 = self._f(slots, step_idx)
            k2 = self._f(slots + 0.5 * dt * k1, step_idx)
            return slots + dt * k2

        raise ValueError(f'Unknown integrator: {self.cfg.integrator}')

"""
Latent evolution operator.

A single, weight-shared operator advances the slot tokens by one step in latent
"time".  It is conditioned on the history-context tokens by cross-attention and
made aware of its position in the rollout by a learned step embedding.

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

from .config import HFM1DConfig
from .modules import SelfAttnBlock, CrossAttnBlock


class _TendencyBlock(nn.Module):
    """One block: slot self-mixing followed by context read-in."""

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.self_blk  = SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.cross_blk = CrossAttnBlock(cfg.d_model, cfg.d_ctx, cfg.n_heads,
                                        cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.cross_blk(self.self_blk(x), context)


class _Tendency(nn.Module):
    """Computes dS = f(S, context): slot self-mixing + context read-in."""

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.blocks = nn.ModuleList([_TendencyBlock(cfg) for _ in range(cfg.n_evo_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.d_model)
        # Zero-init → identity integrator at start of training
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, slots: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = slots
        for blk in self.blocks:
            x = blk(x, context)
        return self.head(self.norm(x))


class EvolutionOperator(nn.Module):
    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        self.tendency  = _Tendency(cfg)
        self.step_emb  = nn.Embedding(cfg.max_rollout, cfg.d_model)
        nn.init.trunc_normal_(self.step_emb.weight, std=0.02)
        # Learnable latent step size, initialised to dt = 1.
        self.log_dt = nn.Parameter(torch.zeros(()))

    def _f(self, slots: torch.Tensor, context: torch.Tensor,
           step_idx: int) -> torch.Tensor:
        idx = torch.tensor(min(step_idx, self.cfg.max_rollout - 1), device=slots.device)
        return self.tendency(slots + self.step_emb(idx), context)

    def forward(self, slots: torch.Tensor, context: torch.Tensor,
                step_idx: int) -> torch.Tensor:
        """Advance slots by one latent step.  slots: [B, n_slots, d_model]."""
        dt = torch.exp(self.log_dt)

        if self.cfg.integrator == 'euler':
            return slots + dt * self._f(slots, context, step_idx)

        if self.cfg.integrator == 'rk2':          # midpoint
            k1 = self._f(slots, context, step_idx)
            k2 = self._f(slots + 0.5 * dt * k1, context, step_idx)
            return slots + dt * k2

        raise ValueError(f'Unknown integrator: {self.cfg.integrator}')

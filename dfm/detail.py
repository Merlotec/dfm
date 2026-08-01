"""Phase-2 detail generation: the closure, moved out of the autoencoder.

The AE is now pure deterministic transport (warp + gain).  Everything transport
cannot do -- creating structure that is not in X_0 -- lives here, split by what
kind of thing it is:

    r  =  E[r | resolved]      +   (r - E[r | resolved])
          "-- MeanHead ------"      "-- FlowHead (sampled) ----"
          deterministic             stochastic (subgrid)

Why here and not in the AE.  The AE is teacher-forced: it encodes (X_{s-1}, X_s),
so a detail head there is handed its own target and learns to DECODE it rather
than predict it -- that is the memorisation we measured (train 0.008 / val 0.157).
Here both heads condition on the ROLLED-OUT resolved frame, which at prediction
time is all the model has, so there is no answer to memorise and the heads are
forced to generalise.  It also keeps the AE latent purely advective, hence
smooth and predictable for evo.

Temporal coherence.  The deterministic part is a function of the resolved state,
so it inherits the state's coherence for free.  The stochastic part would flicker
if resampled independently each frame (white-in-time is unphysical), so its noise
SEED is carried through the rollout as an AR(1)/Ornstein-Uhlenbeck process --
the standard construction for stochastic turbulence/weather closures.  The seed
has memory; the realised pixels are still regenerated each step, so nothing
unstable is fed back into the physics state.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DFMConfig
from .modules import CrossAttnBlock, sincos_2d
from .warp import timestep_embedding


class _CondTrunk(nn.Module):
    """Shared conditioning stem: resolved frame (+mask) -> [B, r*r, d] tokens.

    Spatially resolved, NOT pooled to a vector: a residual has to land in the
    right place, so every query is conditioned on the local resolved content at
    its own position.
    """

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        r, d, C = cfg.warp_detail_res, cfg.d_model, cfg.in_channels
        self.cfg = cfg
        self.register_buffer('pos', sincos_2d(r, r, d).unsqueeze(0), persistent=False)
        self.frame_embed = nn.Conv2d(C, d, kernel_size=1)
        self.mask_embed  = nn.Conv2d(1, d, kernel_size=1)   # geometry drives structure
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, resolved: torch.Tensor,
                pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r = self.cfg.warp_detail_res
        B = resolved.shape[0]
        f = F.adaptive_avg_pool2d(resolved, (r, r))
        q = self.query.expand(B, r * r, -1) + self.pos
        q = q + self.frame_embed(f).flatten(2).transpose(1, 2)
        if pixel_mask is not None:
            m = pixel_mask[:, :1].to(resolved.dtype)
            if m.shape[0] != B:
                m = m.expand(B, -1, -1, -1)
            m = F.adaptive_avg_pool2d(m, (r, r))
            q = q + self.mask_embed(m).flatten(2).transpose(1, 2)
        return q


class MeanHead(nn.Module):
    """Deterministic closure: E[residual | resolved state].

    Trained with plain L2 -- which is CORRECT here: the L2 minimiser IS the
    conditional mean, so this branch is meant to be smooth.  It is the
    structure-creation term (wakes forming, boundary layers developing): a real
    deterministic function of the resolved field that transport cannot express
    because the content is not in X_0 to be moved.
    """

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        d, C = cfg.d_model, cfg.in_channels
        self.trunk = _CondTrunk(cfg)
        self.state_proj = nn.Linear(d, d)                     # evo latent -> KV
        self.layers = nn.ModuleList([
            CrossAttnBlock(d, d, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.warp_head_layers)
        ])
        self.head = nn.Linear(d, C)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)                        # no-op at init

    def forward(self, resolved: torch.Tensor, state: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None,
                out_hw: Optional[Tuple[int, int]] = None,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.cfg
        r = cfg.warp_detail_res
        H, W = out_hw if out_hw is not None else resolved.shape[-2:]
        q = self.trunk(resolved, pixel_mask)
        # regime conditioning: subgrid statistics depend on Re/viscosity, not just
        # on the resolved frame, so the context tokens join the KV set
        kv = self.state_proj(state)
        if context is not None:
            kv = torch.cat([kv, self.state_proj(context)], dim=1)
        for layer in self.layers:
            q = layer(q, kv)
        out = self.head(q).permute(0, 2, 1).reshape(-1, cfg.in_channels, r, r)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return cfg.warp_detail_range * torch.tanh(out)


class FlowHead(nn.Module):
    """Stochastic closure: samples (residual - mean) by conditional flow matching.

    Target is the zero-mean FLUCTUATION, which is easier and better conditioned
    than the full residual.  Loss is MSE on the flow velocity (score matching),
    so sampling recovers the distribution instead of collapsing to the mean the
    way an L1/L2 regressor on a stochastic target must.
    """

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        d, C = cfg.d_model, cfg.in_channels
        self.trunk = _CondTrunk(cfg)
        self.state_proj = nn.Linear(d, d)
        self.noisy_embed = nn.Conv2d(C, d, kernel_size=1)     # current noisy fluctuation
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([
            CrossAttnBlock(d, d, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.warp_head_layers)
        ])
        self.head = nn.Linear(d, C)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _velocity(self, x_t: torch.Tensor, t: torch.Tensor, resolved: torch.Tensor,
                  state: torch.Tensor, pixel_mask: Optional[torch.Tensor],
                  context: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.cfg
        r = cfg.warp_detail_res
        q = self.trunk(resolved, pixel_mask)
        q = q + self.noisy_embed(x_t).flatten(2).transpose(1, 2)
        q = q + self.time_mlp(timestep_embedding(t, cfg.d_model))[:, None, :]
        kv = self.state_proj(state)
        if context is not None:
            kv = torch.cat([kv, self.state_proj(context)], dim=1)
        for layer in self.layers:
            q = layer(q, kv)
        return self.head(q).permute(0, 2, 1).reshape(-1, cfg.in_channels, r, r)

    def _coarse_w(self, pixel_mask, B, device, dtype):
        r = self.cfg.warp_detail_res
        if pixel_mask is None:
            return torch.ones(B, 1, r, r, device=device, dtype=dtype)
        m = pixel_mask[:, :1].to(dtype)
        if m.shape[0] != B:
            m = m.expand(B, -1, -1, -1)
        return F.adaptive_avg_pool2d(m, (r, r))

    def flow_loss(self, resolved: torch.Tensor, state: torch.Tensor,
                  fluctuation: torch.Tensor,
                  pixel_mask: Optional[torch.Tensor] = None,
                  context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """‖v_θ(x_t, t) - (x1 - x0)‖² with x1 = fluctuation, x0 ~ N(0,I)."""
        r = self.cfg.warp_detail_res
        B = resolved.shape[0]
        x1 = F.adaptive_avg_pool2d(fluctuation, (r, r))
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        tb = t[:, None, None, None]
        x_t = (1 - tb) * x0 + tb * x1
        v = self._velocity(x_t, t, resolved, state, pixel_mask, context)
        w = self._coarse_w(pixel_mask, B, x1.device, x1.dtype).float()
        se = (v.float() - (x1 - x0).float()) ** 2
        return (se * w).sum() / w.expand_as(se).sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, resolved: torch.Tensor, state: torch.Tensor,
               pixel_mask: Optional[torch.Tensor] = None,
               out_hw: Optional[Tuple[int, int]] = None,
               noise: Optional[torch.Tensor] = None,
               n_steps: Optional[int] = None,
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Integrate dx/dt = v_θ from t=0 (noise) to t=1 (fluctuation).

        `noise` is the AR-correlated seed for this frame -- pass the carried state
        so consecutive frames share correlated noise and the detail is coherent in
        time.  None -> independent draw (white in time; flickers).
        """
        cfg = self.cfg
        r = cfg.warp_detail_res
        H, W = out_hw if out_hw is not None else resolved.shape[-2:]
        n_steps = n_steps or cfg.warp_flow_steps
        B = resolved.shape[0]
        x = (noise if noise is not None else
             torch.randn(B, cfg.in_channels, r, r, device=resolved.device,
                         dtype=resolved.dtype))
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=resolved.device, dtype=resolved.dtype)
            x = x + dt * self._velocity(x, t, resolved, state, pixel_mask, context)
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)


class DetailGenerator(nn.Module):
    """Both closure heads + the AR(1) noise process that keeps detail coherent.

    add(resolved, state) -> resolved + mean + fluctuation, with the stochastic
    seed advanced by  z <- rho*z + sqrt(1-rho^2)*eps  each call, so z stays
    exactly N(0,I) (the marginal the flow head was trained on) while gaining a
    correlation time of about -1/ln(rho) frames.  rho is a raw learnable
    parameter squashed to (0,1); it can also be pinned from data.
    """

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        self.mean_head = MeanHead(cfg)
        self.flow_head = FlowHead(cfg)
        # sigmoid(raw) = rho; default raw chosen so rho ~ 0.9 (corr time ~10 frames)
        self.rho_raw = nn.Parameter(torch.tensor(float(cfg.detail_noise_rho_init)))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_raw)

    def init_noise(self, B: int, device, dtype) -> torch.Tensor:
        r = self.cfg.warp_detail_res
        return torch.randn(B, self.cfg.in_channels, r, r, device=device, dtype=dtype)

    def advance_noise(self, z: torch.Tensor) -> torch.Tensor:
        """AR(1) step; preserves the N(0,I) marginal exactly."""
        rho = self.rho.to(z.dtype)
        return rho * z + (1.0 - rho ** 2).sqrt() * torch.randn_like(z)

    def losses(self, resolved: torch.Tensor, state: torch.Tensor,
               target: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None,
               grad_checkpoint: bool = False,
               context: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mean_loss, flow_loss, mean_residual) for one step.

        resolved/target are grad-free w.r.t. the AE (it is frozen).  The mean head
        is detached out of the fluctuation target so the two heads do not fight:
        the flow head models the residual AROUND whatever the mean head predicts.

        grad_checkpoint recomputes each head in backward instead of keeping its
        activations: both heads run r*r (=4096 at detail_res 64) query tokens at
        d_model, on EVERY one of the K rollout steps, and all of it is retained for
        BPTT -- that product, not the full-res fields, is what dominates phase-2
        memory (~4.8 GB of the two heads at B=32,K=6 vs ~0.8 GB for transport).
        """
        r_full = target - resolved
        hw = r_full.shape[-2:]
        if grad_checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            r_mean = checkpoint(
                lambda a, b: self.mean_head(a, b, pixel_mask, out_hw=hw, context=context),
                resolved, state, use_reentrant=False)
            fluct = r_full - r_mean.detach()
            flow_loss = checkpoint(
                lambda a, b, c: self.flow_head.flow_loss(a, b, c, pixel_mask, context),
                resolved, state, fluct, use_reentrant=False)
            w = pixel_mask[:, :1].to(r_full.dtype) if pixel_mask is not None else None
            se = (r_mean.float() - r_full.float()) ** 2
            mean_loss = ((se * w).sum() / w.expand_as(se).sum().clamp_min(1.0)
                         if w is not None else se.mean())
            return mean_loss, flow_loss, r_mean
        r_mean = self.mean_head(resolved, state, pixel_mask, out_hw=r_full.shape[-2:],
                                context=context)
        w = pixel_mask[:, :1].to(r_full.dtype) if pixel_mask is not None else None
        se = (r_mean.float() - r_full.float()) ** 2
        if w is not None:
            mean_loss = (se * w).sum() / w.expand_as(se).sum().clamp_min(1.0)
        else:
            mean_loss = se.mean()
        fluct = (r_full - r_mean.detach())
        flow_loss = self.flow_head.flow_loss(resolved, state, fluct, pixel_mask, context)
        return mean_loss, flow_loss, r_mean

    @torch.no_grad()
    def add(self, resolved: torch.Tensor, state: torch.Tensor,
            pixel_mask: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None,
            stochastic: bool = True,
            context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """resolved + deterministic detail [+ sampled fluctuation]."""
        hw = resolved.shape[-2:]
        out = resolved + self.mean_head(resolved, state, pixel_mask, out_hw=hw,
                                        context=context)
        if stochastic:
            out = out + self.flow_head.sample(resolved, state, pixel_mask,
                                              out_hw=hw, noise=noise, context=context)
        return out

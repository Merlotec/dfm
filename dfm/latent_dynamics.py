"""
Latent dynamics model for the BPTT-free two-phase design (phase 2).

Given a frozen LatentAutoencoder (phase 1), we encode ground-truth latents
relative to a fixed anchor X_0:

    L_t = encode(X_0, X_t)          (t = 0 .. n, all anchored to the same X_0)

and train a latent transformer to advance the latent by one step, conditioned on
the history context:

    L_{t+1} ≈ Dynamics(L_t, context, t)

The targets L_{t+1} are *precomputed* from ground-truth frames and detached, so
training is single-step teacher forcing — no rollout and no backprop-through-time.
Only the dynamics operator and the context encoder are trained; the AE is frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional

from .config import DFMConfig
from .evolution import EvolutionOperator
from .context_encoder import ContextEncoder
from .encoder import FrameEncoder
from .autoencoder import LatentAutoencoder
from .modules import add_relative_noise


class LatentDynamics(nn.Module):
    """
    Autoregressive latent predictor: (L_t, context, s_0, t) → L_{t+1}.

    Because L_t is a *delta* code relative to the anchor X_0 (and L_0 is ~null),
    the operator also needs the anchor's absolute state to compute the next
    increment.  `state_encoder` produces s_0 = encode(X_0); it is concatenated
    with the run-context so the evolution cross-attention reads both
    "which dynamics" (context) and "current state" (s_0).
    """

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        self.operator       = EvolutionOperator(cfg)     # evolves the delta latent L_t
        self.state_encoder  = FrameEncoder(cfg)          # s = encode(frame)
        self.state_proj     = (nn.Linear(cfg.d_model, cfg.d_ctx)
                               if cfg.d_model != cfg.d_ctx else nn.Identity())
        # Second stream: evolves the state embedding s_t in latent (evolve_state mode),
        # so the state conditioning stays fresh without a decode-based re-anchor.
        self.state_dynamics = EvolutionOperator(cfg)

    def encode_state_raw(self, x: torch.Tensor,
                         pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full-state tokens s = encode(frame)  [B, n_slots, d_model]."""
        B, _, H, W = x.shape
        if pixel_mask is not None:
            xm = x * pixel_mask
            mask_ch = pixel_mask.float().expand(B, 1, H, W)
        else:
            xm = x
            mask_ch = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        return self.state_encoder(torch.cat([xm, mask_ch], dim=1))

    def project_state(self, s_raw: torch.Tensor) -> torch.Tensor:
        return self.state_proj(s_raw)                             # d_model → d_ctx

    def encode_state(self, x: torch.Tensor,
                     pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """State tokens projected to d_ctx (for the delta operator's cross-attn)."""
        return self.state_proj(self.encode_state_raw(x, pixel_mask))

    def evolve_state(self, s_raw: torch.Tensor, context: torch.Tensor,
                     step_idx) -> torch.Tensor:
        """s_t → s_{t+1} in latent (raw d_model space).  step_idx: int or [B] tensor."""
        return self.state_dynamics(s_raw, context, step_idx)

    def forward(self, latent: torch.Tensor, context: torch.Tensor,
                state: torch.Tensor, step_idx) -> torch.Tensor:
        combined = torch.cat([context, state], dim=1)   # [B, K + n_slots, d_ctx]
        return self.operator(latent, combined, step_idx)


class LatentDynamicsTrainer:
    """
    Trains LatentDynamics (+ ContextEncoder) on a frozen AE's latents.

    Load the phase-1 AE checkpoint first (`load_ae`), then call `step(frames)`:
    the AE encodes the per-step latent targets (no grad), and the dynamics model
    is supervised to match the next latent — teacher forced, no BPTT.
    """

    def __init__(
        self,
        cfg: DFMConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        clip_grad: float = 1.0,
        total_steps: Optional[int] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        self.cfg             = cfg
        self.ae              = LatentAutoencoder(cfg)     # frozen (load_ae)
        self.context_encoder = ContextEncoder(cfg)
        self.dynamics        = LatentDynamics(cfg)
        self.pixel_mask      = pixel_mask

        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()

        params = list(self.dynamics.parameters()) + list(self.context_encoder.parameters())
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps or 1_000_000)
        self.clip_grad = clip_grad
        self.global_step = 0

        # Cache of frozen-AE target latents, keyed by window index (fp16, CPU).
        # The AE is frozen, so L_t = encode(X_0, X_t) is fixed — encode once (epoch 0),
        # reuse thereafter, skipping the per-step pair-encodes entirely.
        self.cache_latents = True
        self._latent_cache: dict = {}

    def to(self, device: torch.device) -> "LatentDynamicsTrainer":
        self.ae              = self.ae.to(device)
        self.context_encoder = self.context_encoder.to(device)
        self.dynamics        = self.dynamics.to(device)
        return self

    def load_ae(self, path: str):
        """Load the frozen phase-1 autoencoder weights."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.ae.load_state_dict(ckpt['ae'], strict=False)
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()

    def _encode_targets(self, x0: torch.Tensor, frames: List[torch.Tensor],
                        pixel_mask: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """L_t = encode(X_0, frames[t]) for every t, detached (teacher targets)."""
        latents = []
        with torch.no_grad():
            for xt in frames:
                latents.append(self.ae.encode(x0, xt, pixel_mask).detach())
        return latents

    def _cached_targets(self, x0, pred_frames, pixel_mask, index) -> List[torch.Tensor]:
        """L_t targets, served from the frozen-AE cache when available."""
        if index is None or not self.cache_latents:
            return self._encode_targets(x0, pred_frames, pixel_mask)
        device = x0.device
        idxs = index.tolist()
        pred_len = len(pred_frames)
        if all(i in self._latent_cache for i in idxs):
            cached = [self._latent_cache[i] for i in idxs]            # each [pred_len, K, d] fp16 cpu
            return [torch.stack([c[t] for c in cached]).to(device).float() for t in range(pred_len)]
        latents = self._encode_targets(x0, pred_frames, pixel_mask)   # list_t of [B, K, d]
        for bi, i in enumerate(idxs):
            self._latent_cache[i] = torch.stack(
                [latents[t][bi] for t in range(pred_len)]).half().cpu()
        return latents

    def step(self, context_frames: List[torch.Tensor],
             pred_frames: List[torch.Tensor],
             pixel_mask: Optional[torch.Tensor] = None,
             index: Optional[torch.Tensor] = None) -> float:
        """
        context_frames : history frames  → ContextEncoder → context
        pred_frames    : [X_0, X_1, ..., X_n]  (anchor + future)
        index          : per-window global indices (enables the frozen-AE latent cache)

        Returns the mean single-step latent MSE.
        """
        self.context_encoder.train()
        self.dynamics.train()
        device_type = pred_frames[0].device.type
        amp = device_type == 'cuda'

        x0 = pred_frames[0]
        # anchored latent sequence  L_0 (= encode(X_0,X_0)) .. L_n  (cached; detached)
        latents = self._cached_targets(x0, pred_frames, pixel_mask, index)
        n = len(latents) - 1

        self.optimizer.zero_grad()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            context = self.context_encoder(context_frames, pixel_mask=pixel_mask)

            # denoising regularizer: feed a noised L_t but supervise against the clean
            # L_{t+1}, so the operator learns to correct off-manifold drift at rollout.
            #
            # The n teacher-forced steps are independent (every input is a precomputed
            # target), so they run as ONE batched operator call with the timestep folded
            # into the batch dim (t-major: rows are [t0·B, t1·B, ...]).  This replaces the
            # per-step Python loop's n kernel launches with a single large one.
            ns    = self.cfg.latent_noise_std
            B     = x0.shape[0]
            steps = torch.arange(n, device=x0.device).repeat_interleave(B)     # [n·B]
            L_in  = add_relative_noise(torch.cat(latents[:n], dim=0), ns)       # noised L_0..L_{n-1}
            L_tgt = torch.cat(latents[1:n + 1], dim=0)                          # clean  L_1..L_n
            ctx_rep = context.repeat(n, 1, 1)                                   # same context per step

            if self.cfg.evolve_state:
                # per-step full-state encodings (with grad → trains the encoder via the
                # delta path); detached copies are the teacher targets for state evolution.
                s_raw  = [self.dynamics.encode_state_raw(f, pixel_mask) for f in pred_frames]
                s_proj = torch.cat([self.dynamics.project_state(s) for s in s_raw[:n]], dim=0)
                dp = self.dynamics(L_in, ctx_rep, s_proj, steps)               # L_t → L_{t+1}
                delta_loss = F.mse_loss(dp.float(), L_tgt.float())
                s_in  = torch.cat([s.detach() for s in s_raw[:n]],     dim=0)
                s_tgt = torch.cat([s.detach() for s in s_raw[1:n + 1]], dim=0)
                sp = self.dynamics.evolve_state(s_in, ctx_rep, steps)          # s_t → s_{t+1}
                state_loss = F.mse_loss(sp.float(), s_tgt.float())
                loss = delta_loss + self.cfg.state_loss_weight * state_loss
            else:
                state = self.dynamics.encode_state(x0, pixel_mask)             # fixed anchor s_0
                pred  = self.dynamics(L_in, ctx_rep, state.repeat(n, 1, 1), steps)
                loss  = F.mse_loss(pred.float(), L_tgt.float())

        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            self._advance()
            return float('nan')

        loss.backward()
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(
                list(self.dynamics.parameters()) + list(self.context_encoder.parameters()),
                self.clip_grad,
            )
        self.optimizer.step()
        self._advance()
        return loss.item()

    def _advance(self):
        self.scheduler.step()
        self.global_step += 1

    @torch.no_grad()
    def rollout(self, context_frames: List[torch.Tensor], x0: torch.Tensor,
                n_steps: int, reencode_every: int = 0,
                pixel_mask: Optional[torch.Tensor] = None,
                n_active_slots: Optional[int] = None) -> List[torch.Tensor]:
        """
        Inference: roll the latent forward and decode.  Latent starts at
        L_0 = encode(X_0, X_0) (zero evolution); every `reencode_every` steps the
        anchor is reset to the latest prediction (0 = never).

        `n_active_slots` (ordered-slot hierarchy only): decode from just the first
        N slots — the compute/quality dial that a nested latent enables.
        """
        self.ae.eval(); self.context_encoder.eval(); self.dynamics.eval()
        context = self.context_encoder(context_frames, pixel_mask=pixel_mask)
        slot_w  = (self.ae.slot_mask.hard(x0.shape[0], n_active_slots, x0.device)
                   if n_active_slots is not None else None)
        anchor  = x0
        latent  = self.ae.encode(anchor, anchor, pixel_mask)
        s_raw   = self.dynamics.encode_state_raw(anchor, pixel_mask)  # s_0 for this anchor
        preds: List[torch.Tensor] = []
        for i in range(n_steps):
            if self.cfg.evolve_state:
                s_raw = self.dynamics.evolve_state(s_raw, context, i)  # evolve state in latent (no decode)
            state  = self.dynamics.project_state(s_raw)
            latent = self.dynamics(latent, context, state, i)
            pred   = self.ae.decode(anchor, latent, pixel_mask, slot_w)
            if pixel_mask is not None:
                pred = pred * pixel_mask
            preds.append(pred)
            if reencode_every > 0 and (i + 1) % reencode_every == 0 and i + 1 < n_steps:
                # decode-based re-anchor: refreshes detail, delta budget, and re-syncs s
                anchor = pred
                latent = self.ae.encode(anchor, anchor, pixel_mask)
                s_raw  = self.dynamics.encode_state_raw(anchor, pixel_mask)
        return preds

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None) -> float:
        self.context_encoder.eval(); self.dynamics.eval()
        device = next(self.dynamics.parameters()).device
        total, count = 0.0, 0
        for context_b, pred_b in dataloader:
            context_frames = [context_b[:, t].to(device) for t in range(context_b.shape[1])]
            pred_frames    = [pred_b[:, t].to(device)    for t in range(pred_b.shape[1])]
            x0 = pred_frames[0]
            latents = self._encode_targets(x0, pred_frames, pixel_mask)
            context = self.context_encoder(context_frames, pixel_mask=pixel_mask)
            n = len(latents) - 1
            if self.cfg.evolve_state:
                states = [self.dynamics.encode_state(f, pixel_mask) for f in pred_frames]
            else:
                s0 = self.dynamics.encode_state(x0, pixel_mask)
                states = [s0] * len(pred_frames)
            loss = 0.0
            for t in range(n):
                pred = self.dynamics(latents[t], context, states[t], t)
                loss += float(F.mse_loss(pred.float(), latents[t + 1].float()))
            total += loss / max(1, n); count += 1
        return total / count if count else float('nan')

    def save(self, path: str):
        def _u(m): return getattr(m, '_orig_mod', m)   # unwrap torch.compile
        torch.save({
            'dynamics':        _u(self.dynamics).state_dict(),
            'context_encoder': _u(self.context_encoder).state_dict(),
            'ae':              _u(self.ae).state_dict(),   # frozen, stored for standalone inference
            'optimizer':       self.optimizer.state_dict(),
            'scheduler':       self.scheduler.state_dict(),
            'cfg':             self.cfg,
            'global_step':     self.global_step,
        }, path)

    def load(self, path: str):
        def _u(m): return getattr(m, '_orig_mod', m)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        _u(self.dynamics).load_state_dict(ckpt['dynamics'], strict=False)
        if 'context_encoder' in ckpt:
            _u(self.context_encoder).load_state_dict(ckpt['context_encoder'], strict=False)
        if 'ae' in ckpt:
            _u(self.ae).load_state_dict(ckpt['ae'], strict=False)
            for p in self.ae.parameters():
                p.requires_grad_(False)
        if 'optimizer' in ckpt:
            try: self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e: print(f'  [load] optimizer not restored ({e})')
        if 'scheduler' in ckpt:
            try: self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception: pass
        self.global_step = ckpt.get('global_step', 0)

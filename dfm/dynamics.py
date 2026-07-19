"""Phase 2: autonomous rollout of the increment latents.

The frozen phase-1 AE defines the representation: increment latents
L_s = encode(X_{s-1}, X_s) (velocity-like; transport + detail streams).  The
evolution transformer learns to advance them WITHOUT frames:

    L_0 = encode(X_0, X_0)          # content-bearing, zero-motion seed
    L̂_{s+1} = evo(L̂_s, step=s)      # attention across both streams —
                                     # detail→transport = backscatter

Decoding accumulates the predicted increments (warp.compose) and applies the
composite to X_0 once per step (decoder.step).  Loss = teacher latent matching
(‖L̂_s − L_s‖, weight cfg.latent_loss_weight) + decoded-field loss through the
accumulator — multi-step credit flows through the whole rollout (BPTT over the
horizon; horizons are short).
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .autoencoder import LatentAutoencoder
from .config import DFMConfig
from .evolution import EvolutionOperator
from .losses import FluidLoss
from .warp import identity_map


class RolloutTrainer:
    """Trains the EvolutionOperator on a frozen AE."""

    def __init__(self, cfg: DFMConfig, lr: float = 3e-4, weight_decay: float = 1e-5,
                 clip_grad: float = 1.0, l1_weight: float = 0.1,
                 total_steps: Optional[int] = None,
                 pixel_mask: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.ae = LatentAutoencoder(cfg)
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.evo = EvolutionOperator(cfg)
        self.criterion = FluidLoss(l1_weight, pixel_mask=pixel_mask)
        self.clip_grad = clip_grad
        self.opt = optim.AdamW(self.evo.parameters(), lr=lr,
                               weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_steps or 100_000)
        self.global_step = 0

    def to(self, device: torch.device) -> "RolloutTrainer":
        self.ae = self.ae.to(device)
        self.evo = self.evo.to(device)
        self.criterion = self.criterion.to(device)
        return self

    def load_ae(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.ae.load_state_dict(ckpt['ae'], strict=False)
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()

    # ---- rollout core ---------------------------------------------------------

    def _rollout(self, frames: torch.Tensor, pixel_mask: Optional[torch.Tensor],
                 K: int, training: bool):
        """Returns (field_loss_mean, latent_loss_mean, xhat_last)."""
        cfg = self.cfg
        B, _, C, H, W = frames.shape
        x0 = frames[:, 0]
        x0m = x0 * pixel_mask[:, :1] if pixel_mask is not None else x0

        with torch.no_grad():                       # frozen AE: teachers + seed
            L = self.ae.encode(x0, x0, pixel_mask)  # zero-motion seed
            teachers = [self.ae.encode(frames[:, s - 1], frames[:, s], pixel_mask)
                        for s in range(1, K + 1)]

        D, G = identity_map(B, C, H, W, frames.device, torch.float32)
        field_sum  = frames.new_zeros(())
        latent_sum = frames.new_zeros(())
        xhat = None
        for s in range(1, K + 1):
            L = self.evo(L, step_idx=s - 1)
            latent_sum = latent_sum + F.mse_loss(L, teachers[s - 1])
            xhat, D, G = self.ae.decoder.step(L, D, G, x0m, use_detail=True)
            field_sum = field_sum + self.criterion(xhat, frames[:, s])
        return field_sum / K, latent_sum / K, xhat

    # ---- training / validation ------------------------------------------------

    def step(self, frames: torch.Tensor,
             pixel_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """frames [B, K+1, C, H, W]; rollout length ~ U{horizon_min..horizon_max}
        (capped by the window).  Returns (field_loss, latent_loss)."""
        cfg = self.cfg
        self.evo.train()
        K = min(frames.shape[1] - 1,
                random.randint(cfg.horizon_min, cfg.horizon_max))
        self.opt.zero_grad()
        field, latent, _ = self._rollout(frames, pixel_mask, K, training=True)
        loss = field + cfg.latent_loss_weight * latent
        if not torch.isfinite(loss):
            self.opt.zero_grad()
            self._advance()
            return float('nan'), float('nan')
        loss.backward()
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.evo.parameters(), self.clip_grad)
        self.opt.step()
        self._advance()
        return field.item(), latent.item()

    def _advance(self):
        self.scheduler.step()
        self.global_step += 1

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None) -> float:
        self.evo.eval()
        device = next(self.evo.parameters()).device
        total, count = 0.0, 0
        for _, pred_b in dataloader:
            frames = pred_b.to(device)
            K = min(frames.shape[1] - 1, self.cfg.horizon)
            field, _, _ = self._rollout(frames, pixel_mask, K, training=False)
            total += float(field); count += 1
        return total / count if count else float('nan')

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, n_steps: int,
                pixel_mask: Optional[torch.Tensor] = None) -> list:
        """Inference: X_0 → n_steps decoded frames (no ground truth needed)."""
        self.evo.eval()
        B, C, H, W = x0.shape
        x0m = x0 * pixel_mask[:, :1] if pixel_mask is not None else x0
        L = self.ae.encode(x0, x0, pixel_mask)
        D, G = identity_map(B, C, H, W, x0.device, torch.float32)
        frames = []
        for s in range(n_steps):
            L = self.evo(L, step_idx=s)
            xhat, D, G = self.ae.decoder.step(L, D, G, x0m, use_detail=True)
            frames.append(xhat)
        return frames

    # ---- checkpointing --------------------------------------------------------

    def save(self, path: str):
        torch.save({'evo': self.evo.state_dict(), 'opt': self.opt.state_dict(),
                    'scheduler': self.scheduler.state_dict(), 'cfg': self.cfg,
                    'global_step': self.global_step}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.evo.load_state_dict(ckpt['evo'])
        for name, obj in [('opt', self.opt), ('scheduler', self.scheduler)]:
            if name in ckpt:
                try:
                    obj.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f'  [load] {name} not restored ({e})')
        self.global_step = ckpt.get('global_step', 0)

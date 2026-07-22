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
from dfm.distributed import host_grad_sync_enabled, allreduce_grads, allreduce_stats

from .autoencoder import LatentAutoencoder
from .config import DFMConfig
from .evolution import EvolutionOperator
from .losses import FluidLoss
from .warp import identity_map, masked_source


class RolloutTrainer:
    """Trains the EvolutionOperator on a frozen AE."""

    def __init__(self, cfg: DFMConfig, lr: float = 3e-4, weight_decay: float = 1e-5,
                 clip_grad: float = 1.0, l1_weight: float = 0.1,
                 total_steps: Optional[int] = None,
                 pixel_mask: Optional[torch.Tensor] = None,
                 latent_loss_weight: Optional[float] = None):
        self.cfg = cfg
        # hyperparams.json carries latent_loss_weight in BOTH the model and the
        # training section.  This used to read only cfg (the model value), so the
        # training-section value was dead config -- with 1.0 vs 0.1 that made the
        # latent term ~20x the field term, i.e. a near-pure latent regressor.
        # Explicit argument wins; None falls back to the model-config default.
        self.latent_loss_weight = (cfg.latent_loss_weight if latent_loss_weight is None
                                   else float(latent_loss_weight))
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

    def wrap_ddp(self, device: torch.device):
        from dfm.distributed import wrap_ddp
        self.evo = wrap_ddp(self.evo, device, find_unused_parameters=True)

    def load_ae(self, path: str):
        from .autoencoder import remap_ae_pyramid_keys, strip_compile_prefix
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = remap_ae_pyramid_keys(strip_compile_prefix(ckpt['ae']))
        missing, unexpected = self.ae.load_state_dict(sd, strict=False)
        # strict=False is needed for the pyramid remap, but it also means a genuine
        # architecture mismatch loads NOTHING and trains the dynamics against a
        # randomly-initialised decoder, silently.  Say so instead.
        if missing or unexpected:
            print(f'  [load_ae] WARNING: {len(missing)} missing, {len(unexpected)} '
                  f'unexpected keys -- the frozen AE is only PARTIALLY loaded.')
            for k in list(missing)[:5]:    print(f'    missing:    {k}')
            for k in list(unexpected)[:5]: print(f'    unexpected: {k}')
        else:
            print(f'  [load_ae] all {len(sd)} tensors loaded cleanly')
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
        x0m = masked_source(x0, pixel_mask, cfg.warp_fill_holes)

        with torch.no_grad():                       # frozen AE: teachers + seed
            L = self.ae.encode(x0, x0, pixel_mask)  # zero-motion seed
            teachers = [self.ae.encode(frames[:, s - 1], frames[:, s], pixel_mask)
                        for s in range(1, K + 1)]

        amp = frames.device.type in ('cuda', 'xpu')
        D, G = identity_map(B, C, H, W, frames.device, torch.bfloat16 if amp else torch.float32)
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
        self.last_K = K            # so reference_losses() can match this step's horizon
        self.opt.zero_grad()
        field, latent, _ = self._rollout(frames, pixel_mask, K, training=True)
        loss = field + self.latent_loss_weight * latent
        (bad,) = allreduce_stats(0.0 if torch.isfinite(loss) else 1.0)
        if bad > 0.0:
            self.opt.zero_grad()
            self._advance()
            return float('nan'), float('nan')
        loss.backward()
        if host_grad_sync_enabled():
            allreduce_grads([self.evo])
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.evo.parameters(), self.clip_grad)
        self.opt.step()
        self._advance()
        return field.item(), latent.item()

    def _advance(self):
        self.scheduler.step()
        self.global_step += 1

    @torch.no_grad()
    def reference_losses(self, frames: torch.Tensor,
                         pixel_mask: Optional[torch.Tensor] = None,
                         K: Optional[int] = None) -> Tuple[float, float]:
        """The two reference points that make `field` readable — (teacher_forced, persistence).

        `field` alone is uninterpretable: it is bounded below by the FROZEN AE's own
        reconstruction error, so a flat field loss can mean either "the dynamics
        operator has stopped improving" or "it is already at the decoder's floor".

        teacher_forced — the identical rollout, but decoding the TRUE latents instead
          of evo's predictions.  This IS that floor: the best field loss reachable
          with this AE.  field/teacher_forced ~ 1 means the operator is done and the
          remaining error is the autoencoder's, not the dynamics'.
        persistence   — the do-nothing model (X̂_s = X_0), the phase-2 twin of
          AutoencoderTrainer.persistence_baseline, giving the same readable r/b.

        Costs an extra decode rollout, so call it at logging cadence, not every step.
        """
        cfg = self.cfg
        B, K1, C, H, W = frames.shape
        K = min(K1 - 1, cfg.horizon if K is None else K)
        x0 = frames[:, 0]
        x0m = masked_source(x0, pixel_mask, cfg.warp_fill_holes)
        teachers = [self.ae.encode(frames[:, s - 1], frames[:, s], pixel_mask)
                    for s in range(1, K + 1)]
        amp = frames.device.type in ('cuda', 'xpu')
        D, G = identity_map(B, C, H, W, frames.device,
                            torch.bfloat16 if amp else torch.float32)
        tf_sum   = frames.new_zeros(())
        pers_sum = frames.new_zeros(())
        for s in range(1, K + 1):
            xhat, D, G = self.ae.decoder.step(teachers[s - 1], D, G, x0m, use_detail=True)
            tf_sum   = tf_sum + self.criterion(xhat, frames[:, s])
            pers_sum = pers_sum + self.criterion(x0m, frames[:, s])
        return float(tf_sum / K), float(pers_sum / K)

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[float, float, float]:
        """(field, teacher_forced, persistence) averaged over the loader."""
        self.evo.eval()
        device = next(self.evo.parameters()).device
        total, tf_total, pers_total, count = 0.0, 0.0, 0.0, 0
        for _, pred_b in dataloader:
            frames = pred_b.to(device)
            K = min(frames.shape[1] - 1, self.cfg.horizon)
            field, _, _ = self._rollout(frames, pixel_mask, K, training=False)
            tf, pers = self.reference_losses(frames, pixel_mask, K=K)
            total += float(field); tf_total += tf; pers_total += pers; count += 1
        if not count:
            return float('nan'), float('nan'), float('nan')
        return total / count, tf_total / count, pers_total / count

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, n_steps: int,
                pixel_mask: Optional[torch.Tensor] = None) -> list:
        """Inference: X_0 → n_steps decoded frames (no ground truth needed)."""
        self.evo.eval()
        B, C, H, W = x0.shape
        x0m = masked_source(x0, pixel_mask, self.cfg.warp_fill_holes)
        L = self.ae.encode(x0, x0, pixel_mask)
        amp = x0.device.type in ('cuda', 'xpu')
        D, G = identity_map(B, C, H, W, x0.device, torch.bfloat16 if amp else torch.float32)
        frames = []
        for s in range(n_steps):
            L = self.evo(L, step_idx=s)
            xhat, D, G = self.ae.decoder.step(L, D, G, x0m, use_detail=True)
            frames.append(xhat)
        return frames

    # ---- checkpointing --------------------------------------------------------

    def save(self, path: str):
        # Unwrap torch.compile before serialising, exactly as AutoencoderTrainer.save
        # does -- saving from the compiled handle writes `_orig_mod.`-prefixed keys
        # that will not load into a bare EvolutionOperator (infer.py, or --resume,
        # which restores BEFORE compile is applied).
        def _u(m): return getattr(m, '_orig_mod', m)
        torch.save({'evo': _u(self.evo).state_dict(), 'opt': self.opt.state_dict(),
                    'scheduler': self.scheduler.state_dict(), 'cfg': self.cfg,
                    'global_step': self.global_step}, path)

    def load(self, path: str):
        from .autoencoder import strip_compile_prefix
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.evo.load_state_dict(strip_compile_prefix(ckpt['evo']))
        for name, obj in [('opt', self.opt), ('scheduler', self.scheduler)]:
            if name in ckpt:
                try:
                    obj.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f'  [load] {name} not restored ({e})')
        self.global_step = ckpt.get('global_step', 0)

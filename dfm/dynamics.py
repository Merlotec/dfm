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
from .context import ContextEncoder
from .detail import DetailGenerator
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
        # The closure lives HERE now (not in the AE): conditioned on the rolled-out
        # resolved frame, so it must generalise rather than decode a teacher code.
        self.detail = DetailGenerator(cfg)
        # Regime summary from a decoupled block of frames (dfm/context.py).  Trained
        # here in phase 2, so the frozen phase-1 AE is unaffected.
        self.context_encoder = ContextEncoder(cfg) if cfg.use_context else None
        self.criterion = FluidLoss(l1_weight, pixel_mask=pixel_mask)
        self.clip_grad = clip_grad
        trainable = list(self.evo.parameters()) + list(self.detail.parameters())
        if self.context_encoder is not None:
            trainable += list(self.context_encoder.parameters())
        self.opt = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_steps or 100_000)
        self.global_step = 0
        self.last_field_detail = float('nan')
        self.last_field_sampled = float('nan')
        self.last_mean_loss = 0.0
        self.last_flow_loss = 0.0
        self.last_detail_loss = 0.0
        self.last_detail_on = False

    def to(self, device: torch.device) -> "RolloutTrainer":
        self.ae = self.ae.to(device)
        self.evo = self.evo.to(device)
        self.detail = self.detail.to(device)
        if self.context_encoder is not None:
            self.context_encoder = self.context_encoder.to(device)
        self.criterion = self.criterion.to(device)
        return self

    def wrap_ddp(self, device: torch.device):
        from dfm.distributed import wrap_ddp
        self.evo = wrap_ddp(self.evo, device, find_unused_parameters=True)
        self.detail = wrap_ddp(self.detail, device, find_unused_parameters=True)
        if self.context_encoder is not None:
            self.context_encoder = wrap_ddp(self.context_encoder, device,
                                            find_unused_parameters=True)

    def set_val_mesh_tables(self, masks, fill_index, bbox) -> None:
        """Validation geometries are a DIFFERENT set with their own 0-based ids, so
        they need their own tables -- indexing val ids into the train table would
        silently pick unrelated geometries."""
        self.val_mesh_masks = masks
        self.val_mesh_fill_index = fill_index
        self.val_mesh_bbox = bbox

    def set_mesh_tables(self, masks, fill_index, bbox) -> None:
        """Per-geometry tables (see FVMDataModule.setup / AutoencoderTrainer)."""
        self.mesh_masks = masks
        self.mesh_fill_index = fill_index
        self.mesh_bbox = bbox

    def _mask_ctx(self, pixel_mask, mesh_ids, val: bool = False):
        """(pixel_mask, fill_index, bbox) for this batch -- per-sample when the
        batch mixes geometries, else the shared mask."""
        pre = 'val_mesh_' if val else 'mesh_'
        masks = getattr(self, pre + 'masks', None)
        if mesh_ids is None or masks is None:
            return pixel_mask, None, None
        fidx, bbt = getattr(self, pre + 'fill_index'), getattr(self, pre + 'bbox')
        ids = mesh_ids.to(masks.device).long()
        bb = bbt[ids]
        bbox = (int(bb[:, 0].min()), int(bb[:, 1].max()),
                int(bb[:, 2].min()), int(bb[:, 3].max()))
        return masks[ids], fidx[ids], bbox

    def load_ae(self, path: str):
        from .autoencoder import remap_ae_pyramid_keys, strip_compile_prefix
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = remap_ae_pyramid_keys(strip_compile_prefix(ckpt['ae']))
        missing, unexpected = self.ae.load_state_dict(sd, strict=False)
        # norm_* buffers are expected to be absent in pre-buffer AE checkpoints.
        missing = [k for k in missing if not k.startswith('norm_')]
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
                 K: int, training: bool, fill_index=None, bbox=None, ctx=None,
                 measure_detail: bool = False):
        """Returns (field_loss_mean, latent_loss_mean, xhat_last)."""
        cfg = self.cfg
        B, _, C, H, W = frames.shape
        x0 = frames[:, 0]
        x0m = masked_source(x0, pixel_mask, cfg.warp_fill_holes,
                            cfg.warp_fill_smooth_iters,
                            fill_index=fill_index, bbox=bbox)

        with torch.no_grad():                       # frozen AE: teachers + seed
            L = self.ae.encode(x0, x0, pixel_mask)  # zero-motion seed
            teachers = [self.ae.encode(frames[:, s - 1], frames[:, s], pixel_mask)
                        for s in range(1, K + 1)]

        # regime tokens: encoded ONCE per batch, reused for every rollout step
        context = (self.context_encoder(ctx, pixel_mask)
                   if (self.context_encoder is not None and ctx is not None) else None)

        amp = frames.device.type in ('cuda', 'xpu')
        D, G = identity_map(B, C, H, W, frames.device, torch.bfloat16 if amp else torch.float32)
        field_sum  = frames.new_zeros(())
        latent_sum = frames.new_zeros(())
        mean_sum = frames.new_zeros(())
        flow_sum = frames.new_zeros(())
        # Diagnostic only (no grad): `field` is scored on the TRANSPORT-ONLY frame and
        # the detail heads take a detached input, so no amount of closure improvement
        # can ever move field / f_tf / f_b.  These two measure the frame the model
        # would actually emit.  fd uses the deterministic mean head -- the MMSE
        # estimate, so it is the fair like-for-like against field and tf.  fds adds a
        # sampled fluctuation and SHOULD score worse pointwise (a valid sample is not
        # the conditional mean); judge that one on spectra, not on this number.
        fd_sum  = 0.0
        fds_sum = 0.0
        xhat = None
        # The decoder is deliberately NOT under no_grad: the AE's weights are frozen,
        # but the field loss has to backprop THROUGH it to reach evo's latent.  So the
        # map head's activations are retained -- and it runs sum(res_i^2) query tokens
        # (21504 at 3 pyramid levels) at d_model on EVERY rollout step.  That product
        # is what exhausts device memory in phase 2; recompute it in backward instead.
        def _decode(L_, D_, G_, x0m_):
            return self.ae.decoder.step(L_, D_, G_, x0m_, use_detail=False)

        use_ckpt = training and cfg.grad_checkpoint and torch.is_grad_enabled()
        # warm evo up first: see cfg.detail_start_step
        detail_on = self.global_step >= cfg.detail_start_step
        self.last_detail_on = detail_on
        for s in range(1, K + 1):
            L = self.evo(L, step_idx=s - 1, context=context)
            latent_sum = latent_sum + F.mse_loss(L, teachers[s - 1])
            # AE is pure transport now: this is the RESOLVED (advective) frame.
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                xhat, D, G = checkpoint(_decode, L, D, G, x0m, use_reentrant=False)
            else:
                xhat, D, G = self.ae.decoder.step(L, D, G, x0m, use_detail=False)
            field_sum = field_sum + self.criterion(xhat, frames[:, s],
                                                   pixel_mask=pixel_mask)
            # Closure, conditioned on the rolled-out resolved frame + evo state.
            # detach: the detail heads must not reshape transport or evo (same
            # boosting-style identifiability the old stage A/B split gave us).
            # Skipped entirely before detail_start_step -- not just zero-weighted, so
            # the warmup also costs nothing (the flow head is the expensive part).
            if detail_on:
                ml, fl, _ = self.detail.losses(xhat.detach(), L.detach(),
                                               frames[:, s], pixel_mask,
                                               grad_checkpoint=cfg.grad_checkpoint and training,
                                               context=context)
                mean_sum = mean_sum + ml
                flow_sum = flow_sum + fl
                if measure_detail:
                    with torch.no_grad():
                        det = self.detail.add(xhat.detach(), L.detach(), pixel_mask,
                                              stochastic=False, context=context)
                        fd_sum += float(self.criterion(det, frames[:, s],
                                                       pixel_mask=pixel_mask))
                        # Draw the sampler's noise from a DEDICATED generator: a bare
                        # torch.randn here would consume the global RNG stream and shift
                        # the flow-matching noise of every later rollout step, so a
                        # logging step would train differently from a non-logging one
                        # (measured: max|grad diff| 1.1e-01).  Diagnostics must not
                        # perturb training.
                        dg = getattr(self, '_diag_gen', None)
                        if dg is None:
                            dg = self._diag_gen = torch.Generator().manual_seed(1234)
                        r = cfg.warp_detail_res
                        z = torch.randn(xhat.shape[0], cfg.in_channels, r, r,
                                        generator=dg).to(xhat.device, xhat.dtype)
                        smp = self.detail.add(xhat.detach(), L.detach(), pixel_mask,
                                              stochastic=True, noise=z, context=context)
                        fds_sum += float(self.criterion(smp, frames[:, s],
                                                        pixel_mask=pixel_mask))
        self.last_field_detail = (fd_sum / K) if measure_detail and detail_on else float('nan')
        self.last_field_sampled = (fds_sum / K) if measure_detail and detail_on else float('nan')
        self.last_mean_loss = float(mean_sum.detach() / K)
        self.last_flow_loss = float(flow_sum.detach() / K)
        self.last_detail_loss = (cfg.detail_mean_weight * mean_sum
                                 + cfg.detail_flow_weight * flow_sum) / K
        return field_sum / K, latent_sum / K, xhat

    # ---- training / validation ------------------------------------------------

    def step(self, frames: torch.Tensor,
             pixel_mask: Optional[torch.Tensor] = None,
             mesh_ids: Optional[torch.Tensor] = None,
             ctx: Optional[torch.Tensor] = None,
             measure_detail: bool = False) -> Tuple[float, float]:
        """frames [B, K+1, C, H, W]; rollout length ~ U{horizon_min..horizon_max}
        (capped by the window).  Returns (field_loss, latent_loss)."""
        cfg = self.cfg
        pixel_mask, fill_index, bbox = self._mask_ctx(pixel_mask, mesh_ids)
        self.evo.train()
        self.detail.train()
        if self.context_encoder is not None:
            self.context_encoder.train()
        K = min(frames.shape[1] - 1,
                random.randint(cfg.horizon_min, cfg.horizon_max))
        self.last_K = K            # so reference_losses() can match this step's horizon
        self.opt.zero_grad()
        field, latent, _ = self._rollout(frames, pixel_mask, K, training=True,
                                         fill_index=fill_index, bbox=bbox, ctx=ctx,
                                         measure_detail=measure_detail)
        loss = field + self.latent_loss_weight * latent + self.last_detail_loss
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
                         K: Optional[int] = None,
                         mesh_ids: Optional[torch.Tensor] = None,
                         val: bool = False) -> Tuple[float, float]:
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
        pixel_mask, fill_index, bbox = self._mask_ctx(pixel_mask, mesh_ids, val=val)
        B, K1, C, H, W = frames.shape
        K = min(K1 - 1, cfg.horizon if K is None else K)
        x0 = frames[:, 0]
        x0m = masked_source(x0, pixel_mask, cfg.warp_fill_holes,
                            cfg.warp_fill_smooth_iters,
                            fill_index=fill_index, bbox=bbox)
        teachers = [self.ae.encode(frames[:, s - 1], frames[:, s], pixel_mask)
                    for s in range(1, K + 1)]
        amp = frames.device.type in ('cuda', 'xpu')
        D, G = identity_map(B, C, H, W, frames.device,
                            torch.bfloat16 if amp else torch.float32)
        tf_sum   = frames.new_zeros(())
        pers_sum = frames.new_zeros(())
        for s in range(1, K + 1):
            xhat, D, G = self.ae.decoder.step(teachers[s - 1], D, G, x0m, use_detail=True)
            tf_sum   = tf_sum + self.criterion(xhat, frames[:, s], pixel_mask=pixel_mask)
            pers_sum = pers_sum + self.criterion(x0m, frames[:, s], pixel_mask=pixel_mask)
        return float(tf_sum / K), float(pers_sum / K)

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[float, float, float, float]:
        """(field, teacher_forced, persistence, field_with_detail) over the loader."""
        self.evo.eval()
        self.detail.eval()
        if self.context_encoder is not None:
            self.context_encoder.eval()
        device = next(self.evo.parameters()).device
        total, tf_total, pers_total, fd_total, count = 0.0, 0.0, 0.0, 0.0, 0
        for batch in dataloader:
            ctx_b  = batch[0] if self.context_encoder is not None else None
            pred_b, mesh_b = batch[1], (batch[2] if len(batch) > 2 else None)
            frames = pred_b.to(device)
            if ctx_b is not None:
                ctx_b = ctx_b.to(device)
            K = min(frames.shape[1] - 1, self.cfg.horizon)
            pm, fi, bb = self._mask_ctx(pixel_mask, mesh_b, val=True)
            field, _, _ = self._rollout(frames, pm, K, training=False,
                                        fill_index=fi, bbox=bb, ctx=ctx_b,
                                        measure_detail=True)
            fd = self.last_field_detail
            fd_total += fd if fd == fd else float(field)      # NaN -> closure is off
            tf, pers = self.reference_losses(frames, pixel_mask, K=K, mesh_ids=mesh_b,
                                             val=True)
            total += float(field); tf_total += tf; pers_total += pers; count += 1
        if not count:
            return (float('nan'),) * 4
        return (total / count, tf_total / count, pers_total / count, fd_total / count)

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
        from .autoencoder import atomic_save
        def _u(m): return getattr(m, '_orig_mod', m)
        atomic_save({'evo': _u(self.evo).state_dict(),
                    'detail': _u(self.detail).state_dict(),
                    'context_encoder': (_u(self.context_encoder).state_dict()
                                        if self.context_encoder is not None else None),
                    'opt': self.opt.state_dict(),
                    'scheduler': self.scheduler.state_dict(), 'cfg': self.cfg,
                    'global_step': self.global_step}, path)

    def load(self, path: str):
        from .autoencoder import strip_compile_prefix
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.evo.load_state_dict(strip_compile_prefix(ckpt['evo']))
        if 'detail' in ckpt:
            self.detail.load_state_dict(strip_compile_prefix(ckpt['detail']))
        if ckpt.get('context_encoder') is not None and self.context_encoder is not None:
            self.context_encoder.load_state_dict(
                strip_compile_prefix(ckpt['context_encoder']))
        for name, obj in [('opt', self.opt), ('scheduler', self.scheduler)]:
            if name in ckpt:
                try:
                    obj.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f'  [load] {name} not restored ({e})')
        self.global_step = ckpt.get('global_step', 0)

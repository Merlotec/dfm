"""Phase 1: increment autoencoder for the two-part (convection + closure) model.

    L_s  = encode(X_{s-1}, X_s)          # increment latent (velocity-like)
    d, g = map_head(L_s[:transport])     # small per-step transport map
    D, G = compose(D, G, d, g)           # accumulate: map from X_0
    X̂_s  = apply(X_0, D, G)  [ + DetailHead(L_s[detail:], X̂_s) in stage B ]

Per-step increments are gradient-findable (a pixel or two of true motion);
large displacements are reached by composition, never by search.  Stage A
trains transport alone; stage B freezes it and fits the generative residual —
boosting-style, so generation cannot replace transport.  See config.py for the
full design rationale and warp.py for the map machinery.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List, Tuple
from dfm.distributed import host_grad_sync_enabled, allreduce_grads, allreduce_stats

from .config import DFMConfig
from .modules import (PatchEmbed, LocalSelfAttnBlock, CrossAttnBlock, SelfAttnBlock,
                      sincos_2d, add_relative_noise)
from .discriminator import DFMDiscriminator
from .losses import FluidLoss
from .warp import WarpDecoder, apply_map, compose, gated_recon_loss, identity_map


# ---------------------------------------------------------------------------
# Pair encoder: (X_{s-1}, X_s) → increment latent L_s
# ---------------------------------------------------------------------------

class PairEncoder(nn.Module):
    """Encodes a CONSECUTIVE frame pair into slot latents — the increment code.
    The first cfg.n_transport_slots tokens drive the transport map, the rest the
    closure; both are distilled from the same patch features (their phase-1
    interaction), and the evolution transformer couples them in phase 2."""

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        Ph, Pw = cfg.n_patch_h, cfg.n_patch_w

        # patch-embed [X_{s-1} ; X_s ; mask(s)]  →  2·C + n_mask_ch input channels
        self.patch_embed = PatchEmbed(2 * cfg.in_channels + cfg.n_mask_ch,
                                      cfg.patch_px, cfg.d_model)
        self.register_buffer('pos', sincos_2d(Ph, Pw, cfg.d_model).unsqueeze(0),
                             persistent=False)

        self.layers = nn.ModuleList([
            LocalSelfAttnBlock(cfg.d_model, cfg.n_heads, Ph, Pw, cfg.local_attn_radius,
                               cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.n_enc_layers)
        ])
        self.slots      = nn.Parameter(torch.zeros(1, cfg.n_slots, cfg.d_model))
        self.slot_cross = CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads,
                                         cfg.ae_mlp, cfg.dropout)
        self.slot_layers = nn.ModuleList([
            SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.n_slot_layers)
        ])
        nn.init.trunc_normal_(self.slots, std=0.02)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = xa.shape
        nmc = self.cfg.n_mask_ch
        if pixel_mask is not None:
            valid = pixel_mask[:, :1].float()
            xa = xa * valid
            xb = xb * valid
            mask_ch = pixel_mask.float().expand(B, nmc, H, W)
        else:
            mask_ch = torch.ones(B, nmc, H, W, device=xa.device, dtype=xa.dtype)

        x   = torch.cat([xa, xb, mask_ch], dim=1)
        tok = rearrange(self.patch_embed(x), 'b h w d -> b (h w) d') + self.pos
        for layer in self.layers:
            tok = layer(tok)
        slots = self.slot_cross(self.slots.expand(B, -1, -1), tok)
        for blk in self.slot_layers:
            slots = blk(slots)
        return slots


# ---------------------------------------------------------------------------
# Model: encoder + warp decoder (transport + closure heads)
# ---------------------------------------------------------------------------

class LatentAutoencoder(nn.Module):
    """encode(X_{s-1}, X_s) → L_s ;  decode = warp machinery (see WarpDecoder)."""

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PairEncoder(cfg)
        self.decoder = WarpDecoder(cfg)
        self._init_weights()

    def encode(self, xa, xb, pixel_mask=None):
        return self.encoder(xa, xb, pixel_mask)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # heads must start as exact no-ops: map head → identity map (a fresh model
        # decodes X_0 exactly); detail head → zero residual (stage B starts from
        # stage A's optimum).  The trunc_normal sweep above re-randomised them.
        heads = [self.decoder.map_head.head]
        if self.decoder.detail_head is not None:
            heads.append(self.decoder.detail_head.head)
        for head in heads:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AutoencoderTrainer:
    """Phase-1 trainer: incremental composition over ground-truth sequences.

    step(frames) — frames [B, K+1, C, H, W] = X_0..X_K:
      stage A: transport only (gain freeze curriculum, flow aux, conservation
               prior, optional complexity gate);
      stage B: transport frozen, DetailHead fits the residual, discriminator
               judges the FINAL frame (deepest composite = hardest fake).
    """

    def __init__(self, cfg: DFMConfig, lr: float = 3e-4, weight_decay: float = 1e-5,
                 l1_weight: float = 0.1, clip_grad: float = 1.0,
                 gan_start_step: int = 10_000, gan_ramp_steps: int = 2_000,
                 disc_update_threshold: float = 0.3,
                 total_steps: Optional[int] = None,
                 pixel_mask: Optional[torch.Tensor] = None,
                 norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # (mean, std) per channel — denormalises velocity for the flow aux loss
        self.norm_mean = norm_stats[0].detach().clone().float() if norm_stats else None
        self.norm_std  = norm_stats[1].detach().clone().float() if norm_stats else None
        self.cfg           = cfg
        self.ae            = LatentAutoencoder(cfg)
        self.discriminator = DFMDiscriminator(cfg)
        self.criterion     = FluidLoss(l1_weight, pixel_mask=pixel_mask)
        self.clip_grad     = clip_grad

        self.gen_optimizer  = optim.AdamW(self.ae.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr,
                                         betas=(0.5, 0.999))
        total = total_steps or 100_000
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.gen_optimizer,
                                                              T_max=total)
        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.global_step           = 0

    # ---- plumbing -------------------------------------------------------------

    def to(self, device: torch.device) -> "AutoencoderTrainer":
        self.ae            = self.ae.to(device)
        self.discriminator = self.discriminator.to(device)
        self.criterion     = self.criterion.to(device)
        if self.norm_mean is not None:
            self.norm_mean = self.norm_mean.to(device)
            self.norm_std  = self.norm_std.to(device)
        return self

    def wrap_ddp(self, device: torch.device):
        from dfm.distributed import wrap_ddp
        self.ae            = wrap_ddp(self.ae, device, find_unused_parameters=True)
        self.discriminator = wrap_ddp(self.discriminator, device, find_unused_parameters=True)


    def _advance(self):
        self.scheduler.step()
        self.global_step += 1

    def _adv_weight(self) -> float:
        if self.global_step < self.gan_start_step:
            return 0.0
        ramp = min(1.0, (self.global_step - self.gan_start_step)
                   / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    def _flow_aux_weight(self) -> float:
        cfg = self.cfg
        if (cfg.warp_flow_aux_weight <= 0 or cfg.warp_flow_aux_steps <= 0
                or self.norm_mean is None):
            return 0.0
        return cfg.warp_flow_aux_weight * max(
            0.0, 1.0 - self.global_step / cfg.warp_flow_aux_steps)

    def training_info(self) -> dict:
        cfg = self.cfg
        return {'global_step': self.global_step,
                'adv_weight': self._adv_weight(),
                'stage': 'B' if (cfg.n_detail_slots > 0
                                 and self.global_step >= cfg.warp_stage_a_steps) else 'A'}

    # ---- core sequence pass ---------------------------------------------------

    def _seq_pass(self, frames: torch.Tensor, pixel_mask: Optional[torch.Tensor],
                  training: bool):
        """Returns (recon_report, recon_grad, reg, aux, xhat_last, x_last).

        recon_report = plain criterion (comparable across configs); recon_grad =
        what the optimiser sees (== report unless the complexity gate is active).
        Map math runs in fp32: composite displacement accumulates over steps and
        bf16 position error is visible after warping."""
        cfg = self.cfg
        ae = getattr(self.ae, '_orig_mod', self.ae)
        map_head, detail_head = ae.decoder.map_head, ae.decoder.detail_head
        nt = cfg.n_transport_slots
        stage_b = detail_head is not None and self.global_step >= cfg.warp_stage_a_steps
        B, K1, C, H, W = frames.shape
        K = K1 - 1
        x0m = frames[:, 0] * pixel_mask[:, :1] if pixel_mask is not None else frames[:, 0]

        freeze = training and not stage_b and self.global_step < cfg.warp_gain_freeze_steps
        aux_w = self._flow_aux_weight() if (training and not stage_b) else 0.0
        if aux_w > 0:
            vc = list(cfg.warp_vel_channels)
            v_std  = self.norm_std[vc].view(1, len(vc), 1, 1)
            v_mean = self.norm_mean[vc].view(1, len(vc), 1, 1)
            alpha = map_head.flow_alpha.view(1, 2, 1, 1)
        gate = training and not stage_b and cfg.warp_complexity_gate
        lam_scale = 1.0
        if gate and cfg.warp_gate_anneal_steps > 0:
            lam_scale = max(0.0, 1.0 - self.global_step / cfg.warp_gate_anneal_steps)
            gate = lam_scale > 0.0

        D, G = identity_map(B, C, H, W, frames.device, torch.float32)
        report_sum = frames.new_zeros(())
        grad_sum   = frames.new_zeros(())
        reg_sum    = frames.new_zeros(())
        aux_sum    = frames.new_zeros(())
        xhat = None
        for s in range(1, K1):
            latent = ae.encode(frames[:, s - 1], frames[:, s], pixel_mask)
            if training:
                latent = add_relative_noise(latent, cfg.ae_decode_noise_std)
            d, g = map_head(latent[:, :nt], out_hw=(H, W))
            d, g = d.float(), g.float()
            if stage_b:
                # transport frozen: stage-B loss must not reshape the map — the
                # "cannot replace" guarantee is this detach plus stage A itself
                d, g = d.detach(), g.detach()
            if freeze:
                g = torch.ones_like(g)
            elif not stage_b:
                reg_sum = reg_sum + (g - 1.0).abs().mean()
            if aux_w > 0:
                # backward fetch ≈ upstream: d(p) ≈ −α ⊙ v_phys(p) — physics as
                # scaffolding, annealed away by _flow_aux_weight
                v_phys = frames[:, s, vc] * v_std + v_mean
                aux_sum = aux_sum + ((d + alpha * v_phys) ** 2).mean()
            D, G = compose(D, G, d, g)
            xhat = apply_map(x0m, D, G)
            if stage_b:
                xhat = xhat + detail_head(latent[:, nt:], xhat, out_hw=(H, W)).float()
            step_report = self.criterion(xhat, frames[:, s])
            report_sum = report_sum + step_report
            if gate:
                grad_sum = grad_sum + gated_recon_loss(
                    xhat, frames[:, s], d, g,
                    l1_weight=self.criterion.l1_weight, window=cfg.warp_gate_window,
                    lam_d=cfg.warp_gate_lambda_d * lam_scale,
                    lam_g=cfg.warp_gate_lambda_g * lam_scale,
                    max_disp=map_head.max_disp, gain_range=map_head.gain_range,
                    pixel_mask=pixel_mask)
            else:
                grad_sum = grad_sum + step_report
        return (report_sum / K, grad_sum / K, reg_sum / K, aux_sum / K,
                xhat, frames[:, K])

    # ---- training step --------------------------------------------------------

    def step(self, frames: torch.Tensor,
             pixel_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        cfg = self.cfg
        self.ae.train()
        self.discriminator.train()
        adv_weight = self._adv_weight()
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

        recon, recon_grad, reg, aux, xhat, x_last = self._seq_pass(
            frames, pixel_mask, training=True)
        loss = recon_grad + cfg.warp_gain_l1 * reg + self._flow_aux_weight() * aux
        xhat_m = xhat.float() * pixel_mask[:, :1] if pixel_mask is not None else xhat.float()
        x0 = frames[:, 0]

        def _restore():
            self.gen_optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            for p in self.discriminator.parameters():
                p.requires_grad_(True)

        if adv_weight == 0.0:
            (bad,) = allreduce_stats(0.0 if torch.isfinite(loss) else 1.0)
            if bad > 0.0:
                _restore(); self._advance(); return float('nan'), 0.0
            loss.backward()
            if host_grad_sync_enabled():
                allreduce_grads([self.ae])
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.ae.parameters(), self.clip_grad)
            self.gen_optimizer.step()
            self._advance()
            return recon.item(), 0.0

        # ---- discriminator (stage-B closure GAN, final frame) ----
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        x_last_m = x_last * pixel_mask[:, :1] if pixel_mask is not None else x_last
        real_logit = self.discriminator(x_last_m, x0)
        fake_logit = self.discriminator(xhat_m.detach(), x0)
        d_loss = (F.binary_cross_entropy_with_logits(real_logit,
                                                     torch.full_like(real_logit, 0.9)) +
                  F.binary_cross_entropy_with_logits(fake_logit,
                                                     torch.zeros_like(fake_logit)))
        d_val = d_loss.item()
        (bad_d,) = allreduce_stats(0.0 if math.isfinite(d_val) else 1.0)
        if bad_d > 0.0:
            _restore(); self._advance(); return float('nan'), float('nan')
        disc_healthy = self.disc_update_threshold < d_val < 2.0
        if disc_healthy:
            d_loss.backward()
            if host_grad_sync_enabled():
                allreduce_grads([self.discriminator])
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad)
            self.disc_optimizer.step()
        self.disc_optimizer.zero_grad()

        # ---- generator ----
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        if disc_healthy:
            adv_logit = self.discriminator(xhat_m, x0)
            adv = F.binary_cross_entropy_with_logits(adv_logit,
                                                     torch.ones_like(adv_logit))
            loss = loss + adv_weight * adv
        (bad_g,) = allreduce_stats(0.0 if torch.isfinite(loss) else 1.0)
        if bad_g > 0.0:
            _restore(); self._advance(); return float('nan'), d_val
        loss.backward()
        if host_grad_sync_enabled():
            allreduce_grads([self.ae])
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.ae.parameters(), self.clip_grad)
        self.gen_optimizer.step()
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        self._advance()
        return recon.item(), d_val

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None) -> float:
        self.ae.eval()
        device = next(self.ae.parameters()).device
        total, count = 0.0, 0
        for _, pred_b in dataloader:
            recon, _, _, _, _, _ = self._seq_pass(pred_b.to(device), pixel_mask,
                                                  training=False)
            total += float(recon); count += 1
        return total / count if count else float('nan')

    # ---- checkpointing --------------------------------------------------------

    def save(self, path: str):
        def _u(m): return getattr(m, '_orig_mod', m)
        torch.save({
            'ae':             _u(self.ae).state_dict(),
            'discriminator':  _u(self.discriminator).state_dict(),
            'gen_optimizer':  self.gen_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'scheduler':      self.scheduler.state_dict(),
            'cfg':            self.cfg,
            'global_step':    self.global_step,
        }, path)

    def load(self, path: str):
        def _u(m): return getattr(m, '_orig_mod', m)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        _u(self.ae).load_state_dict(ckpt['ae'], strict=False)
        if 'discriminator' in ckpt:
            _u(self.discriminator).load_state_dict(ckpt['discriminator'], strict=False)
        for name, opt in [('gen_optimizer', self.gen_optimizer),
                          ('disc_optimizer', self.disc_optimizer)]:
            if name in ckpt:
                try:
                    opt.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f'  [load] {name} not restored ({e})')
        if 'scheduler' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e:
                print(f'  [load] scheduler not restored ({e})')
        self.global_step = ckpt.get('global_step', 0)

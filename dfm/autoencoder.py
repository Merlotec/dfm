"""
Latent autoencoder for the BPTT-free two-phase design.

Phase 1 (this file) learns a compact 1-D latent that describes the *evolution*
from an anchor frame X_0 to a later frame X_t:

    L_t   = encode(X_0, X_t)          # PairEncoder
    X_t   = decode(X_0, L_t)          # skip-anchored SlotDecoder

so a fluid state is represented as "anchor + latent" (X_t ≈ X_0 + integral of the
change encoded in L_t).  Trained per (X_0, X_t) pair — no rollout — with a
reconstruction loss plus an adversarial loss.

Phase 2 (see latent_dynamics.py) freezes this AE, encodes ground-truth latents
L_t = encode(X_0, X_t), and trains a latent transformer to predict L_{t+1} from
L_t with teacher forcing — no backprop-through-time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from typing import Optional, Tuple

from .config import HFM1DConfig
from .modules import PatchEmbed, LocalSelfAttnBlock, CrossAttnBlock, SkipEncoder, sincos_2d
from .decoder import SlotDecoder
from .discriminator import HFMDiscriminator
from .trainer import FluidLoss


# ---------------------------------------------------------------------------
# Pair encoder: (X_0, X_t) → L_t
# ---------------------------------------------------------------------------

class PairEncoder(nn.Module):
    """Encodes the concatenated pair (X_0, X_t) into slot latents L_t."""

    pos: torch.Tensor

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.n_patch

        # patch-embed [X_0 ; X_t ; mask]  →  2·C + 1 input channels
        self.patch_embed = PatchEmbed(2 * cfg.in_channels + 1, cfg.patch_px, cfg.d_model)
        self.register_buffer('pos', sincos_2d(P, cfg.d_model).unsqueeze(0), persistent=False)

        self.layers = nn.ModuleList([
            LocalSelfAttnBlock(cfg.d_model, cfg.n_heads, P, cfg.local_attn_radius,
                               cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_enc_layers)
        ])
        self.slots      = nn.Parameter(torch.zeros(1, cfg.n_slots, cfg.d_model))
        self.slot_cross = CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads,
                                         cfg.mlp_ratio, cfg.dropout)
        nn.init.trunc_normal_(self.slots, std=0.02)

    def forward(self, x0: torch.Tensor, xt: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = x0.shape
        if pixel_mask is not None:
            x0 = x0 * pixel_mask
            xt = xt * pixel_mask
            mask_ch = pixel_mask.float().expand(B, 1, H, W)
        else:
            mask_ch = torch.ones(B, 1, H, W, device=x0.device, dtype=x0.dtype)

        x   = torch.cat([x0, xt, mask_ch], dim=1)                    # [B, 2C+1, H, W]
        tok = rearrange(self.patch_embed(x), 'b h w d -> b (h w) d') + self.pos
        for layer in self.layers:
            tok = layer(tok)
        return self.slot_cross(self.slots.expand(B, -1, -1), tok)   # [B, n_slots, d]


# ---------------------------------------------------------------------------
# Latent autoencoder
# ---------------------------------------------------------------------------

class LatentAutoencoder(nn.Module):
    """encode(X_0, X_t) → L_t ;  decode(X_0, L_t) → X_t."""

    def __init__(self, cfg: HFM1DConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder      = PairEncoder(cfg)
        self.skip_encoder = SkipEncoder(cfg.in_channels, cfg.skip_ch)
        self.decoder      = SlotDecoder(cfg)          # decode(L, skip(X_0))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the final synthesis layers → decoder starts as a pass-through
        # of the tent-folded output (residual refinement learned).
        for seq in [self.decoder.synth.head, self.decoder.synth.post_conv]:
            for child in reversed(list(seq.children())):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    break

    def encode(self, x0: torch.Tensor, xt: torch.Tensor,
               pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x0, xt, pixel_mask)

    def decode(self, x0: torch.Tensor, latent: torch.Tensor,
               pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = self.skip_encoder(x0 * pixel_mask if pixel_mask is not None else x0)
        return self.decoder(latent, skip)

    def forward(self, x0: torch.Tensor, xt: torch.Tensor,
                pixel_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x0, xt, pixel_mask)
        return self.decode(x0, latent, pixel_mask), latent


# ---------------------------------------------------------------------------
# Autoencoder trainer (reconstruction + adversarial, per pair — no rollout)
# ---------------------------------------------------------------------------

class AutoencoderTrainer:
    """Trains LatentAutoencoder on (X_0, X_t) pairs with recon + GAN loss."""

    def __init__(
        self,
        cfg: HFM1DConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        l1_weight: float = 0.1,
        gan_start_step: int = 10_000,
        gan_ramp_steps: int = 2_000,
        disc_update_threshold: float = 0.5,
        clip_grad: float = 1.0,
        total_steps: Optional[int] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        self.cfg           = cfg
        self.ae            = LatentAutoencoder(cfg)
        self.discriminator = HFMDiscriminator(cfg)
        self.criterion     = FluidLoss(l1_weight, pixel_mask=pixel_mask)

        self.gen_optimizer  = optim.AdamW(self.ae.parameters(), lr=lr, weight_decay=weight_decay)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(),
                                         lr=cfg.disc_lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optimizer, T_max=total_steps or 1_000_000)

        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.clip_grad             = clip_grad
        self.global_step           = 0

    def _adv_weight(self) -> float:
        if self.global_step < self.gan_start_step:
            return 0.0
        ramp = min(1.0, (self.global_step - self.gan_start_step) / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    def to(self, device: torch.device) -> "AutoencoderTrainer":
        self.ae            = self.ae.to(device)
        self.discriminator = self.discriminator.to(device)
        self.criterion     = self.criterion.to(device)
        return self

    def step(self, x0: torch.Tensor, xt: torch.Tensor,
             pixel_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        self.ae.train()
        self.discriminator.train()
        device_type = x0.device.type
        amp = device_type == 'cuda'
        adv_weight = self._adv_weight()

        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            xhat, _ = self.ae(x0, xt, pixel_mask)
        xhat_masked = xhat.float() * pixel_mask if pixel_mask is not None else xhat.float()

        # discriminator conditions on the anchor X_0 only (constant zero context)
        zero_ctx = torch.zeros(x0.shape[0], 1, self.cfg.d_ctx, device=x0.device)

        def _restore():
            self.gen_optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            for p in self.discriminator.parameters():
                p.requires_grad_(True)

        # ---- reconstruction-only ----
        if adv_weight == 0.0:
            recon = self.criterion(xhat, xt)
            if not torch.isfinite(recon):
                _restore(); self._advance(); return float('nan'), 0.0
            recon.backward()
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.ae.parameters(), self.clip_grad)
            self.gen_optimizer.step()
            self._advance()
            return recon.item(), 0.0

        # ---- discriminator update ----
        for p in self.discriminator.parameters():
            p.requires_grad_(True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            real_logit = self.discriminator(xt,                  x0, zero_ctx)
            fake_logit = self.discriminator(xhat_masked.detach(), x0, zero_ctx)
        d_loss = (F.binary_cross_entropy_with_logits(real_logit, torch.full_like(real_logit, 0.9)) +
                  F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit)))
        d_val = d_loss.item()
        if not math.isfinite(d_val):
            _restore(); self._advance(); return float('nan'), float('nan')
        disc_healthy = self.disc_update_threshold < d_val < 2.0
        if disc_healthy:
            d_loss.backward()
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad)
            self.disc_optimizer.step()
        self.disc_optimizer.zero_grad()

        # ---- generator update ----
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        recon = self.criterion(xhat, xt)
        if disc_healthy:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
                adv_logit = self.discriminator(xhat_masked, x0, zero_ctx)
            adv = F.binary_cross_entropy_with_logits(adv_logit, torch.ones_like(adv_logit))
            total = recon + adv_weight * adv
        else:
            total = recon
        if not torch.isfinite(total):
            _restore(); self._advance(); return float('nan'), d_val
        total.backward()
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.ae.parameters(), self.clip_grad)
        self.gen_optimizer.step()
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        self._advance()
        return recon.item(), d_val

    def _advance(self):
        self.scheduler.step()
        self.global_step += 1

    def training_info(self) -> dict:
        return {'global_step': self.global_step, 'adv_weight': self._adv_weight()}

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None,
                 ae_max_delta: Optional[int] = None) -> float:
        self.ae.eval()
        device = next(self.ae.parameters()).device
        md = ae_max_delta or self.cfg.ae_max_delta
        total, count = 0.0, 0
        for _, pred_b in dataloader:
            n = pred_b.shape[1] - 1
            t = min(md, n)
            x0 = pred_b[:, 0].to(device)
            xt = pred_b[:, t].to(device)
            xhat, _ = self.ae(x0, xt, pixel_mask)
            total += float(self.criterion(xhat, xt)); count += 1
        return total / count if count else float('nan')

    def save(self, path: str):
        def _u(m): return getattr(m, '_orig_mod', m)   # unwrap torch.compile
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
                try: opt.load_state_dict(ckpt[name])
                except Exception as e: print(f'  [load] {name} not restored ({e})')
        if 'scheduler' in ckpt:
            try: self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception: pass
        self.global_step = ckpt.get('global_step', 0)

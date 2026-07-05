"""
Training loop for HFM-1D.

Each step:
  1. ContextEncoder encodes n_context history frames → context [B, K, d_ctx]
  2. HFM1D rolls the current frame forward `horizon` steps → [x̂_1 .. x̂_H]
  3. Reconstruction loss is summed (optionally discounted) over the horizon;
     the adversarial loss is applied per predicted frame once the GAN is active.

The model and context encoder are optimised jointly; gradients flow through the
entire latent rollout (BPTT over the horizon).

GAN curriculum
--------------
Stage 1 (step < gan_start_step): reconstruction only.
Stage 2 (step >= gan_start_step): adv_weight ramps 0 → cfg.disc_adv_weight over
    gan_ramp_steps steps.  The discriminator only updates while its loss lies in
    (disc_update_threshold, 2.0), keeping the adversarial game balanced.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Tuple

from .config import HFM1DConfig
from .model import HFM1D
from .context_encoder import ContextEncoder
from .discriminator import HFMDiscriminator


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def train_step_gan(
    model: HFM1D,
    context_encoder: ContextEncoder,
    discriminator: HFMDiscriminator,
    context_frames: List[torch.Tensor],
    pred_frames: List[torch.Tensor],
    gen_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    criterion: nn.Module,
    horizon: int,
    reencode_every: int,
    horizon_gamma: float = 1.0,
    adv_weight: float = 0.0,
    latent_loss_weight: float = 0.0,
    clip_grad: float = 1.0,
    pixel_mask: Optional[torch.Tensor] = None,
    disc_update_threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    context_frames  → ContextEncoder → context   (sampled from a random in-run
                                                   offset, decoupled from the seed)
    pred_frames[0]                    → x0 (rollout seed)
    pred_frames[1 .. horizon]         → targets

    Returns (mean recon_loss, disc_loss).  disc_loss is 0.0 when adv_weight == 0.
    """
    gen_optimizer.zero_grad()
    disc_optimizer.zero_grad()

    device_type = pred_frames[0].device.type
    amp = device_type == 'cuda'

    x0      = pred_frames[0]
    targets = pred_frames[1: 1 + horizon]
    # ground-truth previous frame for each prediction (teacher conditioning):
    # pred[i] predicts pred_frames[i+1], whose true predecessor is pred_frames[i].
    prevs   = pred_frames[0: horizon]

    # per-step discount weights, normalised to sum to 1
    weights = torch.tensor([horizon_gamma ** i for i in range(horizon)],
                           device=x0.device)
    weights = weights / weights.sum()

    use_latent = latent_loss_weight > 0.0
    slot_seq: List[torch.Tensor] = []
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        context = context_encoder(context_frames, pixel_mask=pixel_mask)
        if use_latent:
            preds, slot_seq = model(x0, context, horizon=horizon,
                                    reencode_every=reencode_every,
                                    pixel_mask=pixel_mask, return_slots=True)
        else:
            preds = model(x0, context, horizon=horizon,
                          reencode_every=reencode_every, pixel_mask=pixel_mask)

    preds_masked = [
        (p.float() * pixel_mask if pixel_mask is not None else p.float())
        for p in preds
    ]

    def _zero_and_restore() -> None:
        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad_(True)

    def _recon() -> torch.Tensor:
        loss = torch.zeros((), device=x0.device)
        for i in range(horizon):
            loss = loss + weights[i] * criterion(preds[i], targets[i])
        return loss

    def _latent() -> torch.Tensor:
        # Consistency: the evolved slots at step i must match the encoder's
        # representation of the true next frame (detached target).  This forces
        # the dynamics into the evolution operator rather than the decoder.
        if not use_latent:
            return torch.zeros((), device=x0.device)
        loss = torch.zeros((), device=x0.device)
        for i in range(horizon):
            with torch.no_grad(), torch.autocast(device_type=device_type,
                                                 dtype=torch.bfloat16, enabled=amp):
                target_slots = model.encode(targets[i], pixel_mask)
            loss = loss + weights[i] * F.mse_loss(slot_seq[i].float(), target_slots.float())
        return loss

    # ---- reconstruction only ----
    if adv_weight == 0.0:
        recon_loss = _recon()
        gen_loss   = recon_loss + latent_loss_weight * _latent()
        if not torch.isfinite(gen_loss):
            _zero_and_restore()
            return float('nan'), 0.0
        gen_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(context_encoder.parameters()), clip_grad
            )
        gen_optimizer.step()
        return recon_loss.item(), 0.0

    # ---- discriminator update (over all horizon frames) ----
    ctx_detach = context.detach()
    for p in discriminator.parameters():
        p.requires_grad_(True)

    d_loss = torch.zeros((), device=x0.device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
        for i in range(horizon):
            real_logit = discriminator(targets[i],                 prevs[i], ctx_detach)
            fake_logit = discriminator(preds_masked[i].detach(),   prevs[i], ctx_detach)
            real_labels = torch.full_like(real_logit, 0.9)
            d_loss = d_loss + weights[i] * (
                F.binary_cross_entropy_with_logits(real_logit, real_labels) +
                F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
            )
    d_loss_val = d_loss.item()

    if not math.isfinite(d_loss_val):
        _zero_and_restore()
        return float('nan'), float('nan')

    disc_healthy = disc_update_threshold < d_loss_val < 2.0
    if disc_healthy:
        d_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
        disc_optimizer.step()
    disc_optimizer.zero_grad()

    # ---- generator update ----
    for p in discriminator.parameters():
        p.requires_grad_(False)

    recon_loss = _recon()
    if disc_healthy:
        adv_loss = torch.zeros((), device=x0.device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            for i in range(horizon):
                adv_logit = discriminator(preds_masked[i], prevs[i], context)
                adv_loss = adv_loss + weights[i] * F.binary_cross_entropy_with_logits(
                    adv_logit, torch.ones_like(adv_logit)
                )
        total_loss = recon_loss + adv_weight * adv_loss
    else:
        total_loss = recon_loss
    total_loss = total_loss + latent_loss_weight * _latent()

    if not torch.isfinite(total_loss):
        _zero_and_restore()
        return float('nan'), d_loss_val

    total_loss.backward()
    if clip_grad > 0:
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(context_encoder.parameters()), clip_grad
        )
    gen_optimizer.step()

    for p in discriminator.parameters():
        p.requires_grad_(True)

    return recon_loss.item(), d_loss_val


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RolloutGANTrainer:
    """Pairs HFM1D + ContextEncoder with HFMDiscriminator over a latent rollout."""

    def __init__(
        self,
        cfg: HFM1DConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        l1_weight: float = 0.1,
        latent_loss_weight: float = 0.0,
        gan_start_step: int = 10_000,
        gan_ramp_steps: int = 2_000,
        disc_update_threshold: float = 0.5,
        clip_grad: float = 1.0,
        total_steps: Optional[int] = None,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        self.cfg             = cfg
        self.model           = HFM1D(cfg)
        self.context_encoder = ContextEncoder(cfg)
        self.discriminator   = HFMDiscriminator(cfg)
        self.criterion       = FluidLoss(l1_weight, pixel_mask=pixel_mask)

        gen_params = list(self.model.parameters()) + list(self.context_encoder.parameters())
        self.gen_optimizer  = optim.AdamW(gen_params, lr=lr, weight_decay=weight_decay)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=cfg.disc_lr, betas=(0.5, 0.999)
        )
        # T_max = whole run so the LR anneals once (avoids the cosine sawtooth).
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.gen_optimizer, T_max=total_steps or 1_000_000)

        self.gan_start_step        = gan_start_step
        self.gan_ramp_steps        = gan_ramp_steps
        self.disc_update_threshold = disc_update_threshold
        self.clip_grad             = clip_grad
        self.latent_loss_weight    = latent_loss_weight
        self.global_step           = 0

    def _current_adv_weight(self) -> float:
        if self.global_step < self.gan_start_step:
            return 0.0
        steps_in = self.global_step - self.gan_start_step
        ramp = min(1.0, steps_in / max(1, self.gan_ramp_steps))
        return self.cfg.disc_adv_weight * ramp

    def to(self, device: torch.device) -> "RolloutGANTrainer":
        self.model           = self.model.to(device)
        self.context_encoder = self.context_encoder.to(device)
        self.discriminator   = self.discriminator.to(device)
        self.criterion       = self.criterion.to(device)
        return self

    def step(self, context_frames: List[torch.Tensor],
             pred_frames: List[torch.Tensor],
             pixel_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        context_frames : list of n_context tensors [B, C, H, W] (decoupled context)
        pred_frames    : list of (1 + horizon) tensors [B, C, H, W] (seed + targets)
        """
        self.model.train()
        self.context_encoder.train()
        self.discriminator.train()

        # Randomise the rollout length ~ Uniform{horizon_min .. horizon_max}, clamped
        # to the number of target frames actually provided (pred_frames = seed + targets).
        hi = min(self.cfg.horizon_max, len(pred_frames) - 1)
        lo = min(self.cfg.horizon_min, hi)
        horizon = int(torch.randint(lo, hi + 1, (1,)).item())

        # Randomise the re-anchor cadence ~ Uniform{reencode_every_min .. reencode_every_max}
        # (0 = never re-anchor within this rollout).
        r_lo, r_hi = self.cfg.reencode_every_min, self.cfg.reencode_every_max
        reencode_every = int(torch.randint(r_lo, r_hi + 1, (1,)).item())

        recon_loss, disc_loss = train_step_gan(
            self.model, self.context_encoder, self.discriminator,
            context_frames, pred_frames,
            self.gen_optimizer, self.disc_optimizer, self.criterion,
            horizon=horizon,
            reencode_every=reencode_every,
            horizon_gamma=self.cfg.horizon_gamma,
            adv_weight=self._current_adv_weight(),
            latent_loss_weight=self.latent_loss_weight,
            clip_grad=self.clip_grad,
            pixel_mask=pixel_mask,
            disc_update_threshold=self.disc_update_threshold,
        )

        self.scheduler.step()
        self.global_step += 1
        return recon_loss, disc_loss

    def training_info(self) -> dict:
        return {
            'global_step': self.global_step,
            'adv_weight':  self._current_adv_weight(),
            'gan_active':  self._current_adv_weight() > 0.0,
        }

    @torch.no_grad()
    def validate(self, dataloader, pixel_mask: Optional[torch.Tensor] = None) -> float:
        self.model.eval()
        self.context_encoder.eval()
        total, count = 0.0, 0
        device = next(self.model.parameters()).device
        for context_b, pred_b in dataloader:
            context_frames = [context_b[:, t].to(device) for t in range(context_b.shape[1])]
            pred_frames    = [pred_b[:, t].to(device)    for t in range(pred_b.shape[1])]
            context = self.context_encoder(context_frames, pixel_mask=pixel_mask)
            preds   = self.model(pred_frames[0], context, pixel_mask=pixel_mask)
            targets = pred_frames[1: 1 + self.cfg.horizon]
            loss = torch.zeros((), device=device)
            for i in range(len(preds)):
                loss = loss + self.criterion(preds[i], targets[i])
            total += float(loss) / len(preds)
            count += 1
        return total / count if count else float('nan')

    @torch.no_grad()
    def predict(self, context_frames: List[torch.Tensor], x0: torch.Tensor,
                horizon: Optional[int] = None,
                pixel_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        self.model.eval()
        self.context_encoder.eval()
        context = self.context_encoder(context_frames, pixel_mask=pixel_mask)
        preds = self.model(x0, context, horizon=horizon, pixel_mask=pixel_mask)
        if pixel_mask is not None:
            preds = [p * pixel_mask for p in preds]
        return preds

    def save(self, path: str):
        torch.save({
            'model':           self.model.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'discriminator':   self.discriminator.state_dict(),
            'gen_optimizer':   self.gen_optimizer.state_dict(),
            'disc_optimizer':  self.disc_optimizer.state_dict(),
            'scheduler':       self.scheduler.state_dict(),
            'cfg':             self.cfg,
            'global_step':     self.global_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f'  [load] model missing keys (fresh init): {missing}')
        if unexpected:
            print(f'  [load] model unexpected keys (ignored): {unexpected}')
        if 'context_encoder' in ckpt:
            self.context_encoder.load_state_dict(ckpt['context_encoder'], strict=False)
        if 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'], strict=False)
        # Optimizer states are parameter-set-specific; tolerate mismatches from
        # architecture changes (e.g. a resumed checkpoint predating a new layer).
        for name, opt in [('gen_optimizer', self.gen_optimizer),
                          ('disc_optimizer', self.disc_optimizer)]:
            if name in ckpt:
                try:
                    opt.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f'  [load] {name} not restored ({e}); continuing with fresh state')
        if 'scheduler' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e:
                print(f'  [load] scheduler not restored ({e})')
        self.global_step = ckpt.get('global_step', 0)

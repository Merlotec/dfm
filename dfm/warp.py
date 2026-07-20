"""Warp decoder: the latent fully determines a transport map; X_0 is only ever
the OPERAND.

    X̂_t = gain_t ⊙ X_0(id + disp_t)

No network in this path processes X_0 — the map generator reads the slots only.
That closes the "decoder does the physics from X_0 + a thin code" loophole
structurally: without access to the initial condition, arbitrary decoder
capacity cannot implement the flow map Φ_t(X_0); the latent is forced to encode
the transport state itself.  Two further properties fall out:

  * Sharpness by construction — warping transports X_0's sharp content (collider
    edges) instead of re-synthesising it spectrally (no Gibbs ringing).
  * Priced novelty — the map is generated at LOW resolution (cfg.warp_map_res)
    and bilinearly upsampled, so the gain channel can only paint smooth content;
    sharp structure can enter the frame exclusively via the warp.

Map family (per-pixel, in normalised [-1,1] coordinates):
  disp  — displacement field, soft-bounded by cfg.warp_max_disp.  Optionally the
          curl of a latent-generated stream function (cfg.warp_divfree), making
          the warp discretely ~divergence-free (incompressible transport prior).
  gain  — SIGNED multiplicative field, gain = 1 + range·tanh(·).  Signed because
          normalised fields cross zero (an exp-gain could never flip sign);
          identity at zero-init so a fresh model decodes X̂ = X_0 exactly.

compose() implements the semigroup structure for incremental rollouts,

    (M2 ∘ M1) X (p) = g2(p) · G1(φ2(p)) · X(φ1(φ2(p)))

i.e. composites are again warp+gains: smooth map fields are resampled per step,
sharp image content only once, at decode.  The per-pair AE (phase 1) uses the
composite map directly — encode(X_0, X_t) already describes cumulative change —
while compose() is the building block for incremental per-step maps in rollout
training, where the latent holds bounded-complexity increments (velocity-like)
and the accumulator holds the unbounded-complexity composite.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DFMConfig
from .modules import CrossAttnBlock, SelfAttnBlock, sincos_2d


# ---------------------------------------------------------------------------
# Functional core: build / apply / compose maps
# ---------------------------------------------------------------------------

def base_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """Identity sampling grid [B, H, W, 2] (x, y order), align_corners=False."""
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device,
                         dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    return F.affine_grid(theta, [B, 1, H, W], align_corners=False)


def _sample(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """grid_sample with the conventions used throughout this module.

    padding: 'border' (replicate — the right physics at domain edges, e.g. the
    inflow boundary), except on MPS which doesn't implement it; local dev on Mac
    falls back to 'zeros' (matches the hole/pad value of masked data)."""
    pad = 'zeros' if field.device.type == 'mps' else 'border'
    return F.grid_sample(field, grid, mode='bilinear', padding_mode=pad,
                         align_corners=False)


def apply_map(x0: torch.Tensor, disp: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """X̂(p) = gain(p) · X_0(p + disp(p)).

    x0 [B,C,H,W];  disp [B,2,H,W] (x,y, normalised coords);  gain [B,C,H,W].
    disp is a BACKWARD map (where each output pixel fetches from).
    """
    B, _, H, W = x0.shape
    grid = base_grid(B, H, W, x0.device, x0.dtype) + disp.permute(0, 2, 3, 1)
    return gain * _sample(x0, grid)


def compose(disp_prev: torch.Tensor, gain_prev: torch.Tensor,
            disp_inc: torch.Tensor, gain_inc: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Composite ∘ increment (increment applied LAST in time, outermost in the
    backward map):

        D_new(p) = d_inc(p) + D_prev(p + d_inc(p))
        G_new(p) = g_inc(p) · G_prev(p + d_inc(p))

    so apply_map(x0, D_new, G_new) ≈ increment applied to the previous decode.
    Only smooth map fields are resampled here; the image is resampled once, at
    apply_map time.
    """
    B, _, H, W = disp_prev.shape
    grid_inc = base_grid(B, H, W, disp_prev.device, disp_prev.dtype) \
        + disp_inc.permute(0, 2, 3, 1)
    disp_new = disp_inc + _sample(disp_prev, grid_inc)
    gain_new = gain_inc * _sample(gain_prev, grid_inc)
    return disp_new, gain_new


def identity_map(B: int, C: int, H: int, W: int, device, dtype
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(disp=0, gain=1) — apply_map(x0, *identity_map(...)) == x0."""
    return (torch.zeros(B, 2, H, W, device=device, dtype=dtype),
            torch.ones(B, C, H, W, device=device, dtype=dtype))


def _curl(psi: torch.Tensor, h: float) -> torch.Tensor:
    """Perpendicular gradient of a stream function: d = (∂ψ/∂y, -∂ψ/∂x).

    psi [B,1,mh,mw], grid spacing h (normalised coords).  Central differences
    with replicate edges → discretely ~divergence-free displacement [B,2,mh,mw].
    """
    p = F.pad(psi, (1, 1, 1, 1), mode='replicate')
    dpsi_dx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) / (2 * h)
    dpsi_dy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) / (2 * h)
    return torch.cat([dpsi_dy, -dpsi_dx], dim=1)          # (d_x, d_y)


# ---------------------------------------------------------------------------
# MapHead: slots → (disp, gain)
# ---------------------------------------------------------------------------

class WarpMapHead(nn.Module):
    """Reads the slot latent through cross-attention at a low-resolution query
    grid and emits the map fields.  Structurally cannot see X_0."""

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        m = cfg.warp_map_res
        C = cfg.in_channels

        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.register_buffer('pos', sincos_2d(m, m, cfg.d_model).unsqueeze(0),
                             persistent=False)            # [1, m·m, d]
        # slot identity + self-mixing before the map readout
        self.rank_embed = nn.Parameter(torch.zeros(1, cfg.n_slots, cfg.d_model))
        self.slot_layers = nn.ModuleList([
            SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.n_slot_layers)
        ])
        self.layers = nn.ModuleList([
            CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.warp_head_layers)
        ])
        # per-STEP maps are deliberately SMALL (the true per-step displacement is
        # a pixel or two — inside the warp gradient's basin); the composite
        # reaches large displacements by composition, never by search.
        self.max_disp   = cfg.warp_max_disp
        self.gain_range = cfg.warp_gain_range

        # flow scale for the velocity-supervised aux loss (trainer-side):
        # d ≈ −α ⊙ v_phys per axis.  FIXED from config by default — a learnable α
        # collapses to the degenerate aux minimum (α→0 makes the target vanish and
        # d→0 satisfies it; observed).  Signs encode render axis orientation.
        # Lives here so it rides state_dict / device moves for free.
        self.flow_alpha = nn.Parameter(torch.tensor(cfg.warp_flow_alpha, dtype=torch.float32),
                                       requires_grad=cfg.warp_flow_alpha_learnable)

        # divfree: 1 stream-fn channel + 2 UNIFORM-TRANSLATION channels + C gains.
        # The translation term exists because a tanh-bounded ψ cannot hold the
        # linear ramp a constant displacement requires — without it the divfree
        # mode cannot represent net translation (freestream advection!) at all;
        # a constant field is itself divergence-free, so exactness is preserved.
        # Mean flow + fluctuation: the classical decomposition, by construction.
        n_map_ch = (3 if cfg.warp_divfree else 2) + C
        self.head = nn.Linear(cfg.d_model, n_map_ch)
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.rank_embed, std=0.02)
        # zero-init the head → disp = 0, gain = 1: a fresh model decodes X_0 exactly
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, slots: torch.Tensor,
                out_hw: Optional[Tuple[int, int]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """slots [B,S,d] → (disp [B,2,H,W], gain [B,C,H,W]) at out_hw (default cfg.img_hw)."""
        cfg = self.cfg
        B = slots.shape[0]
        m = cfg.warp_map_res
        H, W = out_hw if out_hw is not None else cfg.img_hw

        slots = slots + self.rank_embed[:, :slots.shape[1]]
        for blk in self.slot_layers:
            slots = blk(slots)
        q = self.query.expand(B, m * m, -1) + self.pos
        for layer in self.layers:
            q = layer(q, slots)
        raw = self.head(q).permute(0, 2, 1).reshape(B, -1, m, m)   # [B, n_map_ch, m, m]

        trans = None
        if cfg.warp_divfree:
            # ψ amplitude chosen so the finite-difference curl obeys the soft bound:
            # |Δψ| ≤ 2A over spacing h=2/m  →  |curl| ≤ 2A/h = max_disp for A below.
            h = 2.0 / m
            A = self.max_disp * h / 2.0
            psi = A * torch.tanh(raw[:, :1])
            disp_lo = _curl(psi, h)
            # uniform translation (grid-pooled, per sample): the mean-flow mode the
            # bounded ψ cannot express.  Total |d| soft-bounded by 2·max_disp.
            trans = self.max_disp * torch.tanh(raw[:, 1:3].mean(dim=(2, 3)))  # [B,2]
            gain_raw = raw[:, 3:]
        else:
            disp_lo = self.max_disp * torch.tanh(raw[:, :2])
            gain_raw = raw[:, 2:]

        # low-res → full res: bilinear upsampling is the "priced novelty" bottleneck —
        # the gain can only paint content at warp_map_res smoothness.
        disp = F.interpolate(disp_lo, size=(H, W), mode='bilinear', align_corners=False)
        if trans is not None:
            disp = disp + trans.view(B, 2, 1, 1)
        # diagnostics: mean-flow vs local-structure magnitude — makes the
        # broad-then-specialise progression VISIBLE in the training log
        # (|trans| should grow first; |curl| should follow as the gate releases)
        self.last_trans_mag = float(trans.detach().abs().mean()) if trans is not None else 0.0
        self.last_curl_mag  = float(disp_lo.detach().abs().mean())
        gain = 1.0 + self.gain_range * torch.tanh(
            F.interpolate(gain_raw, size=(H, W), mode='bilinear', align_corners=False))
        return disp, gain


class WarpDecoder(nn.Module):
    """Container for the two decode heads + the rollout decode step.

    Owns map_head (transport stream) and detail_head (closure stream) so both
    ride ae.parameters() / checkpoints.  step() is the single decode primitive
    used by phase-1 training, phase-2 rollout and inference alike.
    """

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        self.map_head = WarpMapHead(cfg)
        self.detail_head = DetailHead(cfg) if cfg.n_detail_slots > 0 else None

    def step(self, latent: torch.Tensor, D: torch.Tensor, G: torch.Tensor,
             x0m: torch.Tensor, use_detail: bool = True,
             freeze_transport: bool = False):
        """One rollout decode step.

        latent [B, n_slots, d] (increment code) → compose the new increment onto
        (D, G), decode X̂ from x0m, optionally add the closure residual.
        Returns (xhat, D, G).  freeze_transport detaches the increment (stage B /
        closure training must not reshape the map)."""
        cfg = self.cfg
        H, W = x0m.shape[-2:]
        nt = cfg.n_transport_slots
        d, g = self.map_head(latent[:, :nt], out_hw=(H, W))
        d, g = d.float(), g.float()
        if freeze_transport:
            d, g = d.detach(), g.detach()
        D, G = compose(D, G, d, g)
        xhat = apply_map(x0m, D, G)
        if use_detail and self.detail_head is not None:
            xhat = xhat + self.detail_head(latent[:, nt:], xhat, out_hw=(H, W)).float()
        return xhat, D, G


# ---------------------------------------------------------------------------
# DetailHead: stage-B generative closure (detail tokens → additive residual)
# ---------------------------------------------------------------------------

class DetailHead(nn.Module):
    """Sub-resolution detail generator — the 'closure' half of the two-part model.

    Reads the DETAIL slot stream (transport never sees these tokens at decode)
    and emits an additive residual on top of the convected frame.  Conditioned on
    the convected frame itself via a pooled embedding — leak-free, because that
    frame is already a function of (X_0, latents), and physically right, because
    subgrid content depends on the resolved field.  Zero-init output => exact
    no-op at the stage-A→B boundary: stage B starts from stage A's optimum.

    Trained ONLY in stage B, with the transport frozen (boosting-style residual
    fitting): generation cannot replace transport because during transport's
    training it did not exist, and during its own training transport is fixed.
    """

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.warp_detail_res
        C = cfg.in_channels

        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.register_buffer('pos', sincos_2d(r, r, cfg.d_model).unsqueeze(0),
                             persistent=False)             # [1, r·r, d]
        self.cond_embed = nn.Conv2d(C, cfg.d_model, kernel_size=1)
        self.layers = nn.ModuleList([
            CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.warp_head_layers)
        ])
        self.head = nn.Linear(cfg.d_model, C)
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)                     # residual == 0 at init

    def forward(self, detail_slots: torch.Tensor, cond: torch.Tensor,
                out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """detail_slots [B,Sd,d]; cond [B,C,H,W] (convected frame, grad-free in
        stage B) → residual [B,C,H,W], soft-bounded by cfg.warp_detail_range."""
        cfg = self.cfg
        B = detail_slots.shape[0]
        r = cfg.warp_detail_res
        H, W = out_hw if out_hw is not None else cfg.img_hw

        c = self.cond_embed(F.adaptive_avg_pool2d(cond, (r, r)))       # [B,d,r,r]
        q = self.query.expand(B, r * r, -1) + self.pos + c.flatten(2).transpose(1, 2)
        for layer in self.layers:
            q = layer(q, detail_slots)
        res = self.head(q).permute(0, 2, 1).reshape(B, -1, r, r)
        res = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
        return cfg.warp_detail_range * torch.tanh(res)


# ---------------------------------------------------------------------------
# Residual-gated complexity loss (the "special loss")
# ---------------------------------------------------------------------------

def gated_recon_loss(xhat: torch.Tensor, target: torch.Tensor,
                     disp: torch.Tensor, gain: torch.Tensor, *,
                     l1_weight: float, window: int, lam_d: float, lam_g: float,
                     max_disp: float, gain_range: float,
                     pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pixel loss, amplified where the local MAP is complex:

        L = mean_w[ err_w · (1 + λ_d·var_w(d)/max_disp² + λ_g·|g−1|_w/range) ]

    NOT the bare product err·var — that has a degenerate zero at var→0 with
    arbitrary error.  The modulation form makes complexity pay for itself in
    PROPORTIONAL error reduction, per window: early training (err high
    everywhere) presses the field toward broad uniform flow — the wide-basin,
    findable regime — and each region's tax falls as its own error falls, an
    automatic, spatially-adaptive graduated-non-convexity schedule.  Smoothness
    pressure also pools matching signal across each window, so flat pixels
    inherit displacement from textured neighbours (Horn–Schunck fill-in).
    Normalisations by max_disp²/gain_range make the λs dimensionless.
    """
    err = (xhat - target) ** 2 + l1_weight * (xhat - target).abs()
    if pixel_mask is not None:
        m = pixel_mask[:, :1]
        err = err * m / m.float().mean().clamp_min(1e-6)   # keep scale ≈ masked mean
    w = window
    err_w = F.avg_pool2d(err.mean(1, keepdim=True), w, stride=w, ceil_mode=True)
    var_w = (F.avg_pool2d(disp ** 2, w, stride=w, ceil_mode=True)
             - F.avg_pool2d(disp, w, stride=w, ceil_mode=True) ** 2
             ).mean(1, keepdim=True) / max(max_disp ** 2, 1e-12)
    gdev_w = F.avg_pool2d((gain - 1.0).abs().mean(1, keepdim=True), w, stride=w,
                          ceil_mode=True) / max(gain_range, 1e-12)
    gate = 1.0 + lam_d * var_w + lam_g * gdev_w
    return (err_w * gate).mean()

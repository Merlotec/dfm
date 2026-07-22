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


def build_fill_index(valid: torch.Tensor) -> torch.Tensor:
    """Flat gather index [H*W]: fluid pixels map to themselves, non-fluid pixels to
    their nearest fluid pixel (Chebyshev-nearest; ties -> largest linear index).

    Ghost cells for the BACKWARD map.  `padding_mode='border'` in _sample only
    extrapolates outside the image RECTANGLE; it cannot help a displacement that
    lands in a collider *inside* the frame, because the source image is hard-zeroed
    there (x0 * pixel_mask).  And 0 is not a neutral "no data" value: frames are
    normalised (raw - mean)/std, so 0 IS the dataset mean -- roughly free-stream
    velocity, injected straight into a boundary layer.  Filling with the nearest
    real fluid value is the zero-gradient (Neumann) condition instead.

    The mask is static per geometry, so this is built once and reused for every
    frame; see fill_index_for().
    """
    v = valid.reshape(*valid.shape[-2:]).bool()
    H, W = v.shape
    assert H * W < 2 ** 24, 'linear indices must stay exactly representable in fp32'
    lin   = torch.arange(H * W, device=valid.device, dtype=torch.float32).view(1, 1, H, W)
    known = v.view(1, 1, H, W)
    if not bool(known.any()):
        return lin.reshape(-1).long()          # degenerate: no fluid at all
    out = torch.where(known, lin, torch.full_like(lin, -1.0))
    # Dilate the known region one ring at a time.  Accepting only pixels that
    # become reachable on THIS ring makes the donor Chebyshev-nearest; max_pool
    # pads with -inf, and a pixel's own 3x3 window contains itself, so an
    # all-unknown neighbourhood stays negative and is simply not accepted yet.
    while not bool(known.all()):
        cand  = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
        newly = (~known) & (cand >= 0)
        if not bool(newly.any()):
            break                              # region unreachable from any fluid
        out   = torch.where(newly, cand, out)
        known = known | newly
    return out.reshape(-1).clamp_min(0).long()


# id() is only sound as a cache key while the object is alive, so each entry keeps a
# strong reference to the mask -- that pins it and makes id() reuse impossible.
# Key on the CALLER's mask object, never on a slice of it: `pixel_mask[:, :1]` builds
# a fresh tensor on every call, so keying on the slice missed every time (rebuilding
# the index per window and pinning a new slice each time -- a leak *and* a slowdown).
# Callers therefore have to hand in the same long-lived mask object each call, which
# is how pixel_mask / val_pm / pmask are already set up once and threaded down.
_FILL_CACHE: dict = {}
_FILL_CACHE_MAX = 8          # bounded: a caller passing fresh tensors can't grow it forever


def fill_index_for(pixel_mask: torch.Tensor) -> torch.Tensor:
    """build_fill_index() memoised per mask tensor (train / val / per-mesh masks)."""
    key = (id(pixel_mask), tuple(pixel_mask.shape), str(pixel_mask.device))
    hit = _FILL_CACHE.get(key)
    if hit is None:
        hit = (pixel_mask, build_fill_index(pixel_mask[:, :1]))
        if len(_FILL_CACHE) >= _FILL_CACHE_MAX:
            _FILL_CACHE.pop(next(iter(_FILL_CACHE)))     # evict oldest (insertion order)
        _FILL_CACHE[key] = hit
    return hit[1]


def apply_fill(x: torch.Tensor, fill_index: torch.Tensor) -> torch.Tensor:
    """Gather x through a fill index -> non-fluid pixels take their nearest fluid value."""
    B, C = x.shape[:2]
    idx = fill_index.view(1, 1, -1).expand(B, C, -1)
    return x.reshape(B, C, -1).gather(2, idx).reshape(x.shape)


def masked_source(x0: torch.Tensor, pixel_mask: Optional[torch.Tensor],
                  fill_holes: bool = False) -> torch.Tensor:
    """The X_0 that apply_map fetches from.

    fill_holes=False reproduces the original behaviour exactly (zero out non-fluid);
    True substitutes nearest-fluid ghost values instead.  Fluid pixels are untouched
    either way, and the loss stays masked to fluid, so filled values are never
    supervised -- they only stop the sampler returning a wrong number.
    """
    if pixel_mask is None:
        return x0
    if not fill_holes:
        return x0 * pixel_mask[:, :1]
    return apply_fill(x0, fill_index_for(pixel_mask))


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
    # The composite and the increment do NOT always share a dtype: _seq_pass keeps
    # both in bf16 under autocast, but WarpDecoder.step upcasts the increment with
    # .float() while the composite starts as bf16 (identity_map under amp).  Adding
    # a fp32 increment to a bf16 base_grid promotes grid_inc to fp32, and
    # grid_sample then rejects the bf16 field ("expected BFloat16 but found Float").
    # Promote everything to the wider dtype: same-dtype callers are unaffected,
    # mixed callers keep the higher precision instead of crashing.  Only reachable
    # on cuda/xpu, where amp is on -- which is why CPU/MPS never hit it.
    dt = torch.promote_types(disp_prev.dtype, disp_inc.dtype)
    disp_prev, gain_prev = disp_prev.to(dt), gain_prev.to(dt)
    disp_inc,  gain_inc  = disp_inc.to(dt),  gain_inc.to(dt)
    B, _, H, W = disp_prev.shape
    grid_inc = base_grid(B, H, W, disp_prev.device, dt) \
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

class _MapLevel(nn.Module):
    """One pyramid level: slots -> a div-free displacement at resolution m.

    The base level (is_base) additionally emits the uniform mean-flow term and
    the gain field.  Finer levels emit ONLY a curl residual — no mean flow, no
    gain — so fine scales are pure flow detail and can never generate.
    Zero-init head => a fresh (or freshly-unlocked) level is an exact no-op.
    """

    pos: torch.Tensor

    def __init__(self, cfg: DFMConfig, m: int, max_disp: float, is_base: bool):
        super().__init__()
        self.cfg, self.m, self.max_disp, self.is_base = cfg, m, max_disp, is_base
        C = cfg.in_channels
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.register_buffer('pos', sincos_2d(m, m, cfg.d_model).unsqueeze(0),
                             persistent=False)
        self.layers = nn.ModuleList([
            CrossAttnBlock(cfg.d_model, cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.warp_head_layers)
        ])
        # channels: divfree base = psi(1)+trans(2)+gain(C); divfree fine = psi(1);
        # raw base = disp(2)+gain(C); raw fine = disp(2)
        if cfg.warp_divfree:
            n_ch = (3 + C) if is_base else 1
        else:
            n_ch = (2 + C) if is_base else 2
        self.head = nn.Linear(cfg.d_model, n_ch)
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, slots, out_hw, disp_scale: float = 1.0):
        cfg = self.cfg
        B = slots.shape[0]
        m = self.m
        H, W = out_hw
        # A d-step jump displaces ~d x further than a 1-step one, so the bound has to
        # open up with d or tanh saturates and large deltas simply cannot be expressed.
        max_disp = self.max_disp * disp_scale
        q = self.query.expand(B, m * m, -1) + self.pos
        for layer in self.layers:
            q = layer(q, slots)
        raw = self.head(q).permute(0, 2, 1).reshape(B, -1, m, m)

        trans = None
        gain = None
        if cfg.warp_divfree:
            h = 2.0 / m
            A = max_disp * h / 2.0
            disp_lo = _curl(A * torch.tanh(raw[:, :1]), h)
            if self.is_base:
                trans = max_disp * torch.tanh(raw[:, 1:3].mean(dim=(2, 3)))
                gain_raw = raw[:, 3:]
        else:
            disp_lo = max_disp * torch.tanh(raw[:, :2])
            gain_raw = raw[:, 2:] if self.is_base else None
        disp = F.interpolate(disp_lo, size=(H, W), mode='bilinear', align_corners=False)
        if trans is not None:
            disp = disp + trans.view(B, 2, 1, 1)
        if self.is_base:
            gain = 1.0 + cfg.warp_gain_range * torch.tanh(
                F.interpolate(gain_raw, size=(H, W), mode='bilinear', align_corners=False))
        return disp, gain, trans


class WarpMapHead(nn.Module):
    """Coarse->fine flow PYRAMID.  Total displacement is the sum of per-level
    div-free fields at doubling resolution; the base level also carries mean flow
    and gain.  Structurally cannot see X_0.  A single level (warp_pyramid_levels
    == 1) is identical to the pre-pyramid head."""

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        self.max_disp   = cfg.warp_max_disp          # base level (for gate/aux norm)
        self.gain_range = cfg.warp_gain_range

        # slot identity + self-mixing, shared across levels
        self.rank_embed = nn.Parameter(torch.zeros(1, cfg.n_slots, cfg.d_model))
        self.slot_layers = nn.ModuleList([
            SelfAttnBlock(cfg.d_model, cfg.n_heads, cfg.ae_mlp, cfg.dropout)
            for _ in range(cfg.n_slot_layers)
        ])
        nn.init.trunc_normal_(self.rank_embed, std=0.02)

        # pyramid levels: res doubles, |disp| bound decays (finer = smaller residual)
        self.levels = nn.ModuleList([
            _MapLevel(cfg, m=cfg.warp_map_res * (2 ** i),
                      max_disp=cfg.warp_max_disp * (cfg.warp_level_disp_decay ** i),
                      is_base=(i == 0))
            for i in range(cfg.warp_pyramid_levels)
        ])
        # curriculum: trainer sets this; None => all levels active
        self.n_active_levels: Optional[int] = None

        # flow-aux scale (see trainer); fixed by default.  Lives here for state_dict.
        self.flow_alpha = nn.Parameter(torch.tensor(cfg.warp_flow_alpha, dtype=torch.float32),
                                       requires_grad=cfg.warp_flow_alpha_learnable)
        # diagnostics
        self.last_trans_mag = 0.0
        self.last_curl_mag  = 0.0
        self.last_level_mags: list = []

    def forward(self, slots: torch.Tensor,
                out_hw: Optional[Tuple[int, int]] = None,
                disp_scale: float = 1.0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        H, W = out_hw if out_hw is not None else cfg.img_hw

        slots = slots + self.rank_embed[:, :slots.shape[1]]
        for blk in self.slot_layers:
            slots = blk(slots)

        n_act = self.n_active_levels if self.n_active_levels is not None else len(self.levels)
        n_act = max(1, min(n_act, len(self.levels)))

        disp_total = None
        gain = None
        self.last_level_mags = []
        # differentiable per-level penalty, finer weighted more (i+1), fine levels
        # only — cached for the trainer to add as cfg.warp_level_l2 * this
        self.last_level_penalty = slots.new_zeros(())
        for i, lvl in enumerate(self.levels[:n_act]):
            d, g, trans = lvl(slots, (H, W), disp_scale)
            disp_total = d if disp_total is None else disp_total + d
            if i == 0:
                gain = g
                self.last_trans_mag = float(trans.detach().abs().mean()) if trans is not None else 0.0
                self.last_curl_mag = float((d - (trans.view(-1, 2, 1, 1) if trans is not None else 0.0)
                                            ).detach().abs().mean())
            else:
                self.last_level_penalty = self.last_level_penalty + (i + 1) * (d ** 2).mean()
            self.last_level_mags.append(float(d.detach().abs().mean()))
        return disp_total, gain


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
             freeze_transport: bool = False, disp_scale: float = 1.0):
        """One rollout decode step.

        latent [B, n_slots, d] (increment code) → compose the new increment onto
        (D, G), decode X̂ from x0m, optionally add the closure residual.
        Returns (xhat, D, G).  freeze_transport detaches the increment (stage B /
        closure training must not reshape the map)."""
        cfg = self.cfg
        H, W = x0m.shape[-2:]
        nt = cfg.n_transport_slots
        d, g = self.map_head(latent[:, :nt], out_hw=(H, W), disp_scale=disp_scale)
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

"""DFM — convection + generative closure, in a latent-rollout model.

The model is a two-part decomposition of fluid evolution:

  * CONVECTION — the latent fully determines a transport map (warp + gain);
    X_0 is only ever the operand.  Per-step increments are small (inside the
    warp gradient's basin) and COMPOSED into the map from X_0 (warp.compose):
    the latent holds bounded-complexity increments (velocity-like), the
    accumulator holds the unbounded-complexity composite.
  * CLOSURE — a generative DetailHead paints the sub-resolution residual
    (the chaos transport cannot carry), conditioned on the convected frame.

The slot set splits into two token streams (transport | detail) that interact
only through attention: the encoder's shared distillation in phase 1, and the
evolution transformer in phase 2 — where detail→transport attention is the
backscatter coupling.  Training is staged, boosting-style, so generation cannot
replace transport: stage A trains transport alone (the DetailHead is absent);
stage B freezes transport and fits the residual (GAN lives here — the closure
is distributional, judged by a discriminator, while transport is pure L2).

Phase 1 (autoencoder.py) trains encoder + heads on ground-truth increment
sequences; phase 2 (dynamics.py) trains the evolution transformer to roll the
increment latents forward autonomously.
"""
from dataclasses import dataclass


@dataclass
class DFMConfig:
    # --- data / geometry -------------------------------------------------------
    img_size: int = 256
    patch_px: int = 16
    in_channels: int = 4
    n_mask_ch: int = 1          # mask channels appended to encoder input

    # --- token dims ------------------------------------------------------------
    d_model: int = 256
    n_slots: int = 64           # total latent tokens
    n_detail_slots: int = 16    # last n tokens = detail (closure) stream
    n_heads: int = 8
    mlp_ratio: float = 4.0      # evolution operator FFN ratio
    ae_mlp: float = 4.0         # encoder/head FFN ratio
    dropout: float = 0.0

    # --- encoder (consecutive pair → increment latent) -------------------------
    n_enc_layers: int = 4
    n_slot_layers: int = 2
    local_attn_radius: int = 2

    # --- transport map (see warp.py) -------------------------------------------
    warp_map_res: int = 32       # map generated at this res, bilinearly upsampled
    warp_max_disp: float = 0.06  # per-STEP |disp| bound: a pixel or two — inside
                                 # every matching basin; composition reaches far
    warp_gain_range: float = 0.5 # per-step gain range (composite grows by product)
    warp_divfree: bool = True    # disp = curl(stream fn) + uniform mean flow
    warp_head_layers: int = 2

    # --- flow pyramid (coarse->fine displacement) ------------------------------
    # The displacement is a sum of div-free levels at doubling resolution
    #   res_i = warp_map_res * 2^i ,   i = 0..warp_pyramid_levels-1
    # coarse level carries mean flow + bulk; each finer level ADDS a smaller
    # residual (SpyNet/optical-flow pyramid; sum works because per-step maps are
    # small and curl is linear, so the sum stays div-free).  Mean-flow and GAIN
    # live ONLY on the base level — finer levels are pure flow detail, never
    # generation, keeping the transport/closure split intact at fine scale.
    #   1 = single level (identical to the pre-pyramid model).
    warp_pyramid_levels: int = 1
    warp_level_disp_decay: float = 0.5   # level-i |disp| bound = warp_max_disp * decay^i
    # curriculum: unlock one finer level every N steps (0 = all active at once).
    # Finer heads are zero-init, so an unlock is a smooth no-op that then learns.
    warp_pyramid_unlock_steps: int = 0
    # regularization: L2 on each finer level's displacement, scaled by level index
    # (finer penalized MORE — smooth flow is the default, fine structure earns its
    # place).  0 disables.  Applied in the trainer via map_head.last_level_mags.
    warp_level_l2: float = 0.0

    # --- closure (DetailHead) --------------------------------------------------
    warp_detail_res: int = 64
    grad_checkpoint: bool = True   # recompute the per-step DetailHead in
                                   # backward (stage-B memory; see _seq_pass)
    warp_detail_range: float = 1.0

    # --- staged training -------------------------------------------------------
    # Stage A (step < warp_stage_a_steps): transport only, DetailHead absent.
    # Stage B: transport frozen (stop-grad through the map), closure fits the
    # residual; the discriminator activates here (gan_start_step in train_ae.py
    # should be >= warp_stage_a_steps).
    warp_stage_a_steps: int = 8000

    # --- flow-over-generation learning aids ------------------------------------
    # (1) velocity supervision: the data OBSERVES the flow; an annealed aux loss
    #     ‖d + α·v_phys‖² places each increment inside the right basin.  α is
    #     FIXED (a learnable α collapses to the degenerate α→0 minimum); set it
    #     from solver metadata: α_axis ≈ Δt_frame / half-domain-extent.  Signs
    #     encode render axis orientation.
    warp_flow_aux_weight: float = 0.1
    warp_flow_aux_steps: int = 5000
    warp_vel_channels: tuple = (0, 1)
    warp_flow_alpha: tuple = (0.01, 0.01)
    warp_flow_alpha_learnable: bool = False
    # (2) gain curriculum + conservation prior: gains frozen at 1 early (the only
    #     way to reduce loss is to WARP), then taxed by λ·|gain−1|₁ forever —
    #     advection conserves intensity along characteristics.
    warp_gain_freeze_steps: int = 2000
    warp_gain_l1: float = 0.01
    # (3) residual-gated complexity loss (stage A): per-window pixel error
    #     amplified by local map complexity — complexity must pay for itself in
    #     proportional local error reduction (automatic, spatially-adaptive
    #     coarse-to-fine; see warp.gated_recon_loss).  The anneal releases
    #     chronically-hard regions (turbulent wakes) from the tax.
    warp_complexity_gate: bool = True
    warp_gate_lambda_d: float = 1.0
    warp_gate_lambda_g: float = 0.0
    warp_gate_window: int = 8
    warp_gate_anneal_steps: int = 0

    # --- latent noise (denoising / rollout-stability regularizer) --------------
    ae_decode_noise_std: float = 0.02

    # --- evolution operator (phase 2) ------------------------------------------
    n_evo_layers: int = 4
    integrator: str = 'rk2'      # 'euler' | 'rk2'
    max_rollout: int = 64        # step-embedding table size
    horizon: int = 4             # eval / inference default rollout length
    horizon_min: int = 2         # phase-2 training length ~ U{min..max}
    horizon_max: int = 6
    latent_loss_weight: float = 1.0   # phase 2: ‖L̂ − L‖ teacher term
    ae_max_delta: int = 8
    n_context_frames: int = 1
    frame_mask: bool = True
    warp_incremental: bool = False

    # --- discriminator (stage-B closure GAN) -----------------------------------
    disc_dim: int = 128
    disc_adv_weight: float = 0.05

    # --- derived ---------------------------------------------------------------
    @property
    def img_hw(self):
        return (self.img_size, self.img_size)

    @property
    def n_patch_h(self):
        return self.img_size // self.patch_px

    @property
    def n_patch_w(self):
        return self.img_size // self.patch_px

    @property
    def n_transport_slots(self):
        return self.n_slots - self.n_detail_slots

    def __post_init__(self):
        assert 0 <= self.n_detail_slots < self.n_slots

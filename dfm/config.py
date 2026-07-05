from dataclasses import dataclass


@dataclass
class DFMConfig:
    """
    Configuration for the latent-rollout fluid model (DFM).

    The model encodes a single frame into a compact set of *slot* tokens, rolls
    those slots forward through a weight-shared latent evolution operator, and
    decodes each rolled state back to an image.  Fine spatial detail is supplied
    to the decoder by a shallow skip encoder anchored on the initial frame — the
    evolution stream itself is a pure slot bottleneck.
    """

    # --- image / patch ---
    img_size: int = 256
    in_channels: int = 4
    patch_px: int = 16

    # --- token / slot dimensions ---
    d_model: int = 256        # patch + slot token dim
    n_slots: int = 64         # number of evolution (slot) tokens
    n_heads: int = 8

    # --- encoder (single frame → slots) ---
    n_enc_layers: int = 4     # self-attention depth over patch tokens

    # --- patch-token self-attention (encoder + decoder) ---
    local_attn_radius: int = 1  # each patch attends to its (2r+1)² neighbourhood (r=1 → 3×3)

    # --- latent evolution operator ---
    n_evo_layers: int = 2     # transformer blocks composing one tendency eval
    integrator: str = 'rk2'   # 'euler' | 'rk2' (midpoint)
    max_rollout: int = 64     # size of the step-index embedding table
    reencode_every: int = 0       # nominal re-anchor cadence (eval / validation / inference)
    reencode_every_min: int = 1   # training: re-anchor cadence ~ Uniform{min .. max} (0 = never)
    reencode_every_max: int = 4

    # --- decoder (slots → image): full transformer (cross + self-attention) ---
    n_dec_layers: int = 4     # transformer decoder blocks (cross-attn slots + self-attn patches)
    skip_ch: int = 32         # shallow skip-encoder channels (initial-frame anchor)

    # --- training rollout ---
    horizon: int = 4          # nominal horizon (eval / validation / inference default)
    horizon_min: int = 2      # training: rollout length ~ Uniform{horizon_min .. horizon_max}
    horizon_max: int = 6
    horizon_gamma: float = 1.0  # per-step loss discount (1.0 = uniform)

    # --- input noise (training only) ---
    noise_std: float = 0.05

    # --- transformer ---
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # --- context encoder (history → conditioning) ---
    n_context_frames: int = 5
    ctx_patch_px: int = 16
    d_ctx: int = 256
    n_ctx_tokens: int = 64
    n_ctx_layers: int = 4
    n_ctx_heads: int = 8

    # --- two-phase latent-AE / dynamics (BPTT-free) ---
    ae_max_delta: int = 6     # AE pair (X_0, X_t): t sampled Uniform{1 .. ae_max_delta}
    evolve_state: bool = False    # also evolve the anchor-state embedding s_t in latent
    state_loss_weight: float = 1.0  # weight of the (teacher-forced) state-prediction loss

    # --- ordered / nested slots (Matryoshka-style; variable token count at inference) ---
    # When on, the decoder reads the slots through a per-example monotone weight ramp
    # w_i = clamp(1 - i/c, 0, 1) applied as an additive log-bias on the cross-attention
    # logits (w=0 → true removal).  The random cutoff c front-loads information into the
    # low-index slots, so the latent can be truncated to any width at inference.
    slot_hierarchy: bool = False
    slot_full_prob: float = 0.25   # fraction of steps trained at full width (all slots, w=1)
    slot_cutoff_min: float = 1.0   # min ramp zero-crossing c (>=1 → slot 0 always fully active)

    # --- GAN discriminator ---
    disc_dim: int = 128
    disc_adv_weight: float = 0.02
    disc_lr: float = 1e-4

    # --- memory ---
    gradient_checkpointing: bool = False

    @property
    def n_patch(self) -> int:
        return self.img_size // self.patch_px

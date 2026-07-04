from dataclasses import dataclass


@dataclass
class HFM1DConfig:
    """
    Configuration for the latent-rollout fluid model (HFM-1D).

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

    # --- decoder (slots → image): a renderer, not a physics engine ---
    n_dec_layers: int = 2     # slot-readout layers (cross-attention only, no spatial reasoning)
    n_smooth_layers: int = 2  # slot-blind local self-attention smoothing layers
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

    # --- GAN discriminator ---
    disc_dim: int = 128
    disc_adv_weight: float = 0.02
    disc_lr: float = 1e-4

    # --- memory ---
    gradient_checkpointing: bool = False

    @property
    def n_patch(self) -> int:
        return self.img_size // self.patch_px

"""
End-to-end smoke test: build a tiny HFM-1D and run forward + backward on random
data.  No dataset or renderer required — validates that the whole latent-rollout
pipeline wires together and produces gradients.

    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import HFM1DConfig, RolloutGANTrainer


def main():
    torch.manual_seed(0)

    # Tiny config for a fast CPU check
    cfg = HFM1DConfig(
        img_size=64, patch_px=16, d_model=48, n_slots=8, n_heads=4,
        n_enc_layers=2, n_evo_layers=1, integrator='rk2',
        skip_ch=8, horizon=3, reencode_every=2,
        n_context_frames=2, ctx_patch_px=16, d_ctx=48,
        n_ctx_tokens=8, n_ctx_layers=1, n_ctx_heads=4,
        disc_dim=16, noise_std=0.05,
    )

    B, C, H, W = 2, cfg.in_channels, cfg.img_size, cfg.img_size
    # decoupled context (n_context frames) + prediction window (seed + horizon)
    context_frames = [torch.randn(B, C, H, W) for _ in range(cfg.n_context_frames)]
    pred_frames    = [torch.randn(B, C, H, W) for _ in range(cfg.horizon + 1)]
    pixel_mask = torch.ones(1, 1, H, W, dtype=torch.bool)
    pixel_mask[..., :4, :] = False  # a few 'hole' pixels

    trainer = RolloutGANTrainer(cfg, gan_start_step=0, gan_ramp_steps=1,
                                disc_update_threshold=0.0, pixel_mask=pixel_mask)

    n_params = lambda m: sum(p.numel() for p in m.parameters())
    print(f'HFM1D:          {n_params(trainer.model)/1e3:.1f}K params')
    print(f'ContextEncoder: {n_params(trainer.context_encoder)/1e3:.1f}K params')
    print(f'Discriminator:  {n_params(trainer.discriminator)/1e3:.1f}K params')

    # --- forward shape check ---
    with torch.no_grad():
        ctx = trainer.context_encoder(context_frames, pixel_mask=pixel_mask)
        preds = trainer.model(pred_frames[0], ctx, pixel_mask=pixel_mask)
    assert len(preds) == cfg.horizon, f'expected {cfg.horizon} preds, got {len(preds)}'
    for i, p in enumerate(preds):
        assert p.shape == (B, C, H, W), f'pred {i} wrong shape: {p.shape}'
    print(f'Context tokens: {tuple(ctx.shape)}')
    print(f'Rollout:        {len(preds)} frames, each {tuple(preds[0].shape)}')

    # --- reconstruction-only step (adv off) ---
    trainer.gan_start_step = 10  # force adv_weight = 0
    recon, disc = trainer.step(context_frames, pred_frames, pixel_mask=pixel_mask)
    print(f'\nRecon-only step:  recon={recon:.4f}  disc={disc:.4f}')
    assert recon == recon, 'recon loss is NaN'

    # --- GAN step (adv on) ---
    trainer.gan_start_step = 0
    recon, disc = trainer.step(context_frames, pred_frames, pixel_mask=pixel_mask)
    print(f'GAN step:         recon={recon:.4f}  disc={disc:.4f}  '
          f'adv_w={trainer.training_info()["adv_weight"]:.3f}')
    assert recon == recon, 'recon loss is NaN'

    # --- gradient sanity: evolution operator must receive gradient ---
    grad_norm = sum(
        p.grad.norm().item() for p in trainer.model.evolution.parameters()
        if p.grad is not None
    )
    print(f'Evolution grad norm (sum): {grad_norm:.4e}')
    assert grad_norm > 0, 'evolution operator received no gradient!'

    # --- checkpoint round-trip ---
    ckpt = Path('/tmp/hfm1d_smoke.pt')
    trainer.save(str(ckpt))
    trainer.load(str(ckpt))
    ckpt.unlink()

    print('\n✅ smoke test passed')


if __name__ == '__main__':
    main()

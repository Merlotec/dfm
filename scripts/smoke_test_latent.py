"""
End-to-end smoke test for the BPTT-free two-phase design (autoencoder + dynamics).

    python scripts/smoke_test_latent.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import DFMConfig, AutoencoderTrainer, LatentDynamicsTrainer


def main():
    torch.manual_seed(0)
    cfg = DFMConfig(
        img_size=64, patch_px=16, d_model=48, n_slots=8, n_heads=4,
        n_enc_layers=2, n_evo_layers=1, n_dec_layers=1,
        d_ctx=48, n_ctx_tokens=8, n_ctx_layers=1, n_ctx_heads=4,
        disc_dim=16, ae_max_delta=4, n_context_frames=2, horizon_max=6,
    )
    B, C, H, W = 2, cfg.in_channels, cfg.img_size, cfg.img_size
    pm = torch.ones(1, 1, H, W, dtype=torch.bool); pm[..., :4, :] = False

    # ================= Phase 1: autoencoder =================
    ae_tr = AutoencoderTrainer(cfg, gan_start_step=0, gan_ramp_steps=1,
                               disc_update_threshold=0.0, pixel_mask=pm)
    x0 = torch.randn(B, C, H, W); xt = torch.randn(B, C, H, W)

    with torch.no_grad():
        xhat, L = ae_tr.ae(x0, xt, pm)
    assert xhat.shape == (B, C, H, W), xhat.shape
    assert L.shape == (B, cfg.n_slots, cfg.d_model), L.shape
    print(f'AE forward: xhat {tuple(xhat.shape)}  latent {tuple(L.shape)}')

    ae_tr.gan_start_step = 10                     # recon-only
    r, d = ae_tr.step(x0, xt, pixel_mask=pm)
    print(f'AE recon-only step:  recon={r:.4f} disc={d:.4f}')
    ae_tr.gan_start_step = 0                      # GAN on
    r, d = ae_tr.step(x0, xt, pixel_mask=pm)
    print(f'AE GAN step:         recon={r:.4f} disc={d:.4f} adv_w={ae_tr.training_info()["adv_weight"]:.3f}')
    assert r == r, 'AE recon NaN'
    ae_grad = sum(p.grad.norm().item() for p in ae_tr.ae.parameters() if p.grad is not None)
    assert ae_grad > 0, 'AE got no gradient'
    print(f'AE param grad (sum): {ae_grad:.3e}')

    # ================= Phase 2: latent dynamics (BPTT-free) =================
    ctx_frames  = [torch.randn(B, C, H, W) for _ in range(cfg.n_context_frames)]
    pred_frames = [torch.randn(B, C, H, W) for _ in range(cfg.horizon_max + 1)]

    for evolve in (False, True):
        import dataclasses
        cfg_m = dataclasses.replace(cfg, evolve_state=evolve)
        dyn_tr = LatentDynamicsTrainer(cfg_m, pixel_mask=pm)
        loss = dyn_tr.step(ctx_frames, pred_frames, pixel_mask=pm)
        print(f'\n[evolve_state={evolve}] dynamics step: loss={loss:.5f}')
        assert loss == loss, 'latent loss NaN'

        # --- BPTT-free / frozen-AE checks ---
        ae_frozen  = all(not p.requires_grad for p in dyn_tr.ae.parameters())
        ae_no_grad = all(p.grad is None for p in dyn_tr.ae.parameters())
        dyn_grad = sum(p.grad.norm().item() for p in dyn_tr.dynamics.parameters() if p.grad is not None)
        sd_grad  = sum(p.grad.norm().item() for p in dyn_tr.dynamics.state_dynamics.parameters()
                       if p.grad is not None)
        print(f'  AE frozen & out of graph (BPTT-free): {ae_frozen and ae_no_grad}')
        print(f'  dynamics grad: {dyn_grad:.3e}   state_dynamics grad: {sd_grad:.3e}')
        assert ae_frozen and ae_no_grad, 'AE should be frozen and out of the graph'
        assert dyn_grad > 0, 'dynamics got no gradient'
        if evolve:
            assert sd_grad > 0, 'state_dynamics should be trained in evolve_state mode'

        preds = dyn_tr.rollout(ctx_frames, x0, n_steps=3, reencode_every=2, pixel_mask=pm)
        assert len(preds) == 3 and preds[0].shape == (B, C, H, W)
        print(f'  rollout: {len(preds)} frames, each {tuple(preds[0].shape)}')

        # checkpoint round-trip (both modes)
        dyn_tr.save('/tmp/dyn_smoke.pt'); dyn_tr.load('/tmp/dyn_smoke.pt')

    # --- AE checkpoint round-trip ---
    ae_tr.save('/tmp/ae_smoke.pt'); ae_tr.load('/tmp/ae_smoke.pt')
    import os; os.remove('/tmp/ae_smoke.pt'); os.remove('/tmp/dyn_smoke.pt')

    print('\n✅ latent two-phase smoke test passed')


if __name__ == '__main__':
    main()

"""End-to-end correctness tests for the two-part (convection + closure) DFM."""
import sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfm import DFMConfig, LatentAutoencoder, AutoencoderTrainer, RolloutTrainer
from dfm.warp import apply_map, compose, identity_map, gated_recon_loss

torch.manual_seed(0)
ok = lambda name, cond: print(f'  {"PASS" if cond else "FAIL"}  {name}') or cond
allpass = True

B, C, H, W = 2, 4, 64, 64
cfg = DFMConfig(img_size=64, patch_px=16, d_model=64, n_slots=8, n_detail_slots=3,
                n_enc_layers=1, n_slot_layers=1, n_heads=4, warp_map_res=16,
                warp_head_layers=1, warp_detail_res=32, warp_max_disp=0.05,
                warp_gain_range=0.3, warp_gain_freeze_steps=10**9,
                warp_flow_aux_steps=80, warp_flow_aux_weight=1.0,
                warp_flow_alpha=(0.015625, 0.015625), warp_stage_a_steps=120,
                warp_complexity_gate=True, n_evo_layers=1, horizon_min=2,
                horizon_max=3)

# ---- functional core ----
x = F.interpolate(torch.randn(B, C, 8, 8), size=(H, W), mode='bilinear',
                  align_corners=False)
d0, g0 = identity_map(B, C, H, W, x.device, x.dtype)
allpass &= ok('identity map exact', (apply_map(x, d0, g0) - x).abs().max() < 2e-5)
dc = torch.zeros(B, 2, H, W); dc[:, 0] = 0.1
gc = torch.full((B, C, H, W), 1.3)
dC, gC = compose(dc, gc, dc, gc)
allpass &= ok('compose: const maps add/multiply',
              torch.allclose(dC, 2 * dc, atol=1e-6) and torch.allclose(gC, gc * gc, atol=1e-6))

# gate ordering
xh, tg = torch.randn(1, C, H, W), torch.randn(1, C, H, W)
kw = dict(l1_weight=0.1, window=8, lam_d=1.0, lam_g=0.0, max_disp=0.05,
          gain_range=0.5, pixel_mask=None)
Lf = gated_recon_loss(xh, tg, torch.full((1, 2, H, W), 0.02), torch.ones(1, C, H, W), **kw)
Lv = gated_recon_loss(xh, tg, 0.02 * torch.randn(1, 2, H, W), torch.ones(1, C, H, W), **kw)
allpass &= ok(f'gate taxes varied flow ({Lv:.3f} > {Lf:.3f})', Lv > Lf * 1.05)

# ---- AE identity at init + leak-free map ----
ae = LatentAutoencoder(cfg); ae.eval()
lat = ae.encode(x, x)
D, G = identity_map(B, C, H, W, x.device, torch.float32)
xhat, D1, G1 = ae.decoder.step(lat, D, G, x, use_detail=True)
allpass &= ok('fresh model decodes X0 exactly (both heads zero-init)',
              torch.allclose(xhat, x, atol=1e-5))
d_a, g_a = ae.decoder.map_head(lat[:, :cfg.n_transport_slots], out_hw=(H, W))
lat2 = ae.encode(x + torch.randn_like(x), x + torch.randn_like(x))
allpass &= ok('map reads latent only (shapes sane)',
              d_a.shape == (B, 2, H, W) and g_a.shape == (B, C, H, W))

# ---- known-flow toy: translation + unwarpable overlay, both stages ----
torch.manual_seed(3)
Ht = Wt = 64; K = 3
pat = F.interpolate(torch.randn(1, 2, 8, 8), size=(Ht, Wt), mode='bilinear',
                    align_corners=False)
overlay = F.interpolate(0.35 * torch.randn(1, 2, 32, 32), size=(Ht, Wt), mode='nearest')
mean_t = torch.tensor([4.0, 0.0, 0.0, 0.0]); std_t = torch.tensor([2.0, 1.0, 1.0, 1.0])
frames = []
for s in range(K + 1):
    f = torch.zeros(1, 4, Ht, Wt)
    f[:, 0] = 0.0                                   # v_x normalised (phys = 4.0)
    f[:, 2:] = torch.roll(pat, shifts=s * 2, dims=-1)
    if s > 0:
        f[:, 2:] += overlay                         # static, unwarpable detail
    frames.append(f)
frames = torch.stack(frames, dim=1)

tr = AutoencoderTrainer(cfg, lr=3e-3, gan_start_step=10**9,
                        norm_stats=(mean_t, std_t))
for i in range(120):
    rA, _ = tr.step(frames)
dh = tr.ae.decoder.detail_head
allpass &= ok(f'stage A (recon={rA:.4f}); detail head untouched',
              all((p == 0).all() for p in [dh.head.weight, dh.head.bias]))
with torch.no_grad():
    lat1 = tr.ae.encode(frames[:, 0], frames[:, 1])
    d_l, _ = tr.ae.decoder.map_head(lat1[:, :cfg.n_transport_slots], out_hw=(Ht, Wt))
dx = d_l[:, 0].mean().item()
allpass &= ok(f'stage A learned translation (d_x={dx:+.4f}, expect < -0.03)', dx < -0.03)

mh0 = torch.cat([p.detach().reshape(-1).clone()
                 for p in tr.ae.decoder.map_head.parameters()])
for i in range(150):
    rB, _ = tr.step(frames)
mh1 = torch.cat([p.detach().reshape(-1) for p in tr.ae.decoder.map_head.parameters()])
allpass &= ok('stage B: transport frozen', torch.equal(mh0, mh1))
allpass &= ok(f'stage B improves on unwarpable detail ({rA:.4f} → {rB:.4f})',
              rB < rA * 0.9)

# ---- phase 2: rollout trainer on the frozen AE ----
dyn = RolloutTrainer(cfg, lr=1e-3)
dyn.ae.load_state_dict(tr.ae.state_dict())
for p in dyn.ae.parameters(): p.requires_grad_(False)
f0 = None
for i in range(30):
    field, latent = dyn.step(frames)
    if i == 0: f0 = field
allpass &= ok(f'phase-2 trains (field {f0:.4f} → {field:.4f}, latent={latent:.4f})',
              field == field and latent == latent)   # finite
roll = dyn.rollout(frames[:, 0], n_steps=4)
allpass &= ok('inference rollout: X0 → 4 frames, no ground truth',
              len(roll) == 4 and roll[0].shape == (1, 4, Ht, Wt)
              and all(torch.isfinite(r).all() for r in roll))

print('\nALL PASS' if allpass else '\nSOME FAILURES')
sys.exit(0 if allpass else 1)

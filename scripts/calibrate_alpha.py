"""Measure warp_flow_alpha from the data: fit the optimal uniform backward shift
between consecutive frames, regress against mean physical velocity.

    d ≈ −α ⊙ v_phys   →   α = −shift / v̄   (per axis, sign included)

Run:  python scripts/calibrate_alpha.py [--data DIR] [--pairs N]
"""
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfm.config import DFMConfig
from dfm.data import FVMDataModule, build_renderer, load_pixel_mask
from dfm.warp import apply_map

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'fvm_gen_datasets')
p.add_argument('--pairs', type=int, default=48)
a = p.parse_args()

cfg = DFMConfig()
dm = FVMDataModule(a.data, n_context=1, horizon=4, batch_size=4, num_workers=0,
                   random_context=True)
dm.setup()
renderer = build_renderer(a.data if (a.data / 'shared_mesh.pkl').exists()
                          else next(d for d in a.data.iterdir() if (d / 'shared_mesh.pkl').exists()),
                          cfg.img_hw)
mask_dir = a.data if (a.data / 'shared_mesh.pkl').exists() \
    else next(d for d in a.data.iterdir() if (d / 'shared_mesh.pkl').exists())
pm = load_pixel_mask(mask_dir, renderer, cfg.img_hw, frame_mask=cfg.frame_mask)[:, :1].float()

shifts, vels = [], []
it = iter(dm.train_dataloader())
while len(shifts) < a.pairs:
    _, pred_b = next(it)
    for b in range(pred_b.shape[0]):
        for s in range(1, pred_b.shape[1]):
            xa = pred_b[b:b+1, s-1] * pm
            xb = pred_b[b:b+1, s] * pm
            # fit a single uniform backward shift t: min ‖ xa(p+t) − xb(p) ‖²
            t = torch.zeros(2, requires_grad=True)
            opt = torch.optim.Adam([t], lr=5e-3)
            for _ in range(120):
                opt.zero_grad()
                disp = t.view(1, 2, 1, 1).expand(1, 2, *cfg.img_hw)
                loss = ((apply_map(xa, disp, torch.ones_like(xa)) - xb) * pm).pow(2).mean()
                loss.backward(); opt.step()
            # mean physical velocity of this pair (denormalised)
            vc = list(cfg.warp_vel_channels)
            v = (pred_b[b, s, vc] * dm.std[vc].view(-1, 1, 1)
                 + dm.mean[vc].view(-1, 1, 1))
            vbar = (v * pm[0]).sum(dim=(1, 2)) / pm.sum()
            shifts.append(t.detach().clone()); vels.append(vbar)
            if len(shifts) >= a.pairs: break
        if len(shifts) >= a.pairs: break

S = torch.stack(shifts); V = torch.stack(vels)
# per-axis least squares: S_axis = −α_axis · V_axis  → α = −<S,V>/<V,V>
alpha = -(S * V).sum(0) / (V * V).sum(0).clamp_min(1e-9)
res = (S + alpha * V).pow(2).mean().sqrt()
print(f'fitted over {len(shifts)} consecutive pairs:')
print(f'  mean shift  = ({S[:,0].mean():+.5f}, {S[:,1].mean():+.5f})  (normalised coords/frame)')
print(f'  mean v_phys = ({V[:,0].mean():+.3f}, {V[:,1].mean():+.3f})')
print(f'  alpha       = ({alpha[0]:+.6f}, {alpha[1]:+.6f})   residual={res:.5f}')
print(f'\n  → hyperparams.json:  "warp_flow_alpha": [{alpha[0]:.6f}, {alpha[1]:.6f}]')
print(f'  sanity: per-step displacement ≈ {(alpha.abs()*V.abs().mean(0)).max():.4f} '
      f'(warp_max_disp={cfg.warp_max_disp}; should be same order, below the bound)')

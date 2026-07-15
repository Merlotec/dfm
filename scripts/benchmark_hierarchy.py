"""
Benchmark: hierarchy vs no-hierarchy.

Trains both variants for a fixed number of steps and reports the cost/benefit:
  - params (AE + dynamics)                          — the size overhead
  - training throughput (samples/s)                 — the training cost
  - quality (AE recon, dynamics latent-MSE)         — the accuracy tax at full width
  - inference: rollout ms/step AND MAE at several   — the variable-width dial the
    slot widths {K, K/2, K/4}                          hierarchy is *for*

The two variants differ ONLY in the hierarchy flags (everything else — d_model,
layers, noise, mlp — is held equal), so the deltas isolate the hierarchy.

Data:
  - default: SYNTHETIC random frames (self-contained; isolates compute/params/inference
    from data-loading, and lets you run it anywhere).  Quality numbers are meaningless
    on random data and are omitted.
  - --data DIR: real frames via the FVM pipeline → also reports recon / rollout MAE.

    python scripts/benchmark_hierarchy.py --ae-steps 300 --dyn-steps 300
    python scripts/benchmark_hierarchy.py --data ../data/fvm_gen_datasets --ae-steps 500 --dyn-steps 500
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import DFMConfig, AutoencoderTrainer, LatentDynamicsTrainer

_ROOT       = Path(__file__).resolve().parents[1]
HYPERPARAMS = _ROOT / 'hyperparams.json'

# the flags that ARE the hierarchy — everything else is held equal
HIER   = dict(slot_hierarchy=True,  dynamics_hierarchy=True,  state_hierarchy=True,  n_slot_layers=2)
NOHIER = dict(slot_hierarchy=False, dynamics_hierarchy=False, state_hierarchy=False, n_slot_layers=0)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def params_m(m) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


def make_cfg(base: dict, hier: bool) -> DFMConfig:
    c = dict(base)
    c.update(HIER if hier else NOHIER)
    return DFMConfig(**c)


# ---------------------------------------------------------------------------
# batch sources
# ---------------------------------------------------------------------------

def synth_frame(cfg, B, device):
    C, (H, W) = cfg.in_channels, cfg.img_hw
    return torch.randn(B, C, H, W, device=device)


# ---------------------------------------------------------------------------
# AE: train for N steps, measure throughput + final recon
# ---------------------------------------------------------------------------

def bench_ae(cfg, device, steps, B, warmup, next_pair, pixel_mask):
    tr = AutoencoderTrainer(cfg, gan_start_step=10**9, pixel_mask=pixel_mask).to(device)  # recon-only
    for _ in range(warmup):
        x0, xt = next_pair(B)
        tr.step(x0, xt, pixel_mask=pixel_mask)
    sync(device)
    t0, losses = time.perf_counter(), []
    for _ in range(steps):
        x0, xt = next_pair(B)
        r, _ = tr.step(x0, xt, pixel_mask=pixel_mask)
        losses.append(r)
    sync(device)
    dt = time.perf_counter() - t0
    return tr, dict(params=params_m(tr.ae),
                    samp_s=steps * B / dt,
                    recon=float(np.mean(losses[-min(20, steps):])))


# ---------------------------------------------------------------------------
# dynamics: train for N steps on the frozen AE, measure throughput + latent MSE
# ---------------------------------------------------------------------------

def bench_dyn(cfg, ae, device, steps, B, warmup, next_window, pixel_mask):
    tr = LatentDynamicsTrainer(cfg, pixel_mask=pixel_mask).to(device)
    tr.ae.load_state_dict(ae.state_dict())               # freeze the just-trained AE
    for p in tr.ae.parameters():
        p.requires_grad_(False)
    tr.ae.eval()
    for _ in range(warmup):
        ctx, pred = next_window(B)
        tr.step(ctx, pred, pixel_mask=pixel_mask)
    sync(device)
    t0, losses = time.perf_counter(), []
    for _ in range(steps):
        ctx, pred = next_window(B)
        losses.append(tr.step(ctx, pred, pixel_mask=pixel_mask))
    sync(device)
    dt = time.perf_counter() - t0
    return tr, dict(params=params_m(tr.dynamics) + params_m(tr.context_encoder),
                    samp_s=steps * B / dt,
                    latent_mse=float(np.mean(losses[-min(20, steps):])))


# ---------------------------------------------------------------------------
# inference: rollout ms/step and (real data) MAE at several slot widths
# ---------------------------------------------------------------------------

@torch.no_grad()
def bench_rollout(tr, cfg, device, widths, ctx, x0, gt, n_steps, reencode, reps=3):
    out = {}
    for N in widths:
        na = None if N == cfg.n_slots else N
        for _ in range(2):                                # warmup
            tr.rollout(ctx, x0, n_steps=n_steps, reencode_every=reencode,
                       pixel_mask=None, n_active_slots=na)
        sync(device)
        preds = []
        t0 = time.perf_counter()
        for _ in range(max(1, reps)):
            preds = tr.rollout(ctx, x0, n_steps=n_steps, reencode_every=reencode,
                               pixel_mask=None, n_active_slots=na)
        sync(device)
        ms = (time.perf_counter() - t0) / reps / n_steps * 1e3
        mae = None
        if gt is not None:
            mae = float(torch.stack([(p - g).abs().mean() for p, g in zip(preds, gt)]).mean())
        out[N] = dict(ms_step=ms, mae=mae)
    return out


# ---------------------------------------------------------------------------

def run_variant(name, cfg, device, args, sources):
    print(f'\n===== {name} =====')
    ae_tr, ae_m = bench_ae(cfg, device, args.ae_steps, args.batch_size, args.warmup,
                           sources['pair'], sources['pixel_mask'])
    print(f'  AE       : {ae_m["params"]:.1f}M params  {ae_m["samp_s"]:.0f} samp/s  '
          f'recon={ae_m["recon"]:.4f}')
    dyn_tr, dyn_m = bench_dyn(cfg, ae_tr.ae, device, args.dyn_steps, args.batch_size, args.warmup,
                              sources['window'], sources['pixel_mask'])
    print(f'  dynamics : {dyn_m["params"]:.1f}M params  {dyn_m["samp_s"]:.0f} samp/s  '
          f'latent_mse={dyn_m["latent_mse"]:.5f}')

    # inference dial: hierarchy → sweep widths; no-hier → full only (truncation is OOD)
    K = cfg.n_slots
    widths = [K, K // 2, K // 4] if cfg.dynamics_hierarchy else [K]
    ctx, x0, gt = sources['rollout']()
    roll = bench_rollout(dyn_tr, cfg, device, widths, ctx, x0, gt,
                         args.roll_steps, args.reencode)
    for N in widths:
        r = roll[N]
        mae = f'  MAE={r["mae"]:.4f}' if r['mae'] is not None else ''
        print(f'  rollout N={N:>4}: {r["ms_step"]:.1f} ms/step{mae}')
    return dict(ae=ae_m, dyn=dyn_m, roll=roll, widths=widths)


def main():
    p = argparse.ArgumentParser(description='Benchmark hierarchy vs no-hierarchy')
    p.add_argument('--data', type=Path, default=None, help='real data dir (else synthetic)')
    p.add_argument('--ae-steps',  type=int, default=300)
    p.add_argument('--dyn-steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--roll-steps', type=int, default=10)
    p.add_argument('--reencode',  type=int, default=0, help='rollout re-anchor cadence (0 = pure latent)')
    p.add_argument('--warmup',    type=int, default=3)
    p.add_argument('--img',       type=int, default=None, help='override img_size (smaller = faster bench)')
    args = p.parse_args()

    torch.manual_seed(0)
    device = get_device()
    with open(HYPERPARAMS) as f:
        base = {k: v for k, v in json.load(f)['model'].items() if not k.startswith('_')}
    if args.img:
        base['img_size'], base['img_w'] = args.img, None
    B = args.batch_size or json.load(open(HYPERPARAMS))['training'].get('batch_size', 4)
    print(f'Device: {device}   batch={B}   ae_steps={args.ae_steps}  dyn_steps={args.dyn_steps}')

    # --- build the two configs (identical except the hierarchy flags) ---
    cfg_no  = make_cfg(base, hier=False)
    cfg_hi  = make_cfg(base, hier=True)

    # --- data sources ---
    if args.data is not None:
        from dfm.data import FVMDataModule, build_renderer, load_pixel_mask
        dm = FVMDataModule(args.data, n_context=cfg_hi.n_context_frames, horizon=cfg_hi.horizon_max,
                           batch_size=B, num_workers=0, random_context=True, return_index=False)
        dm.setup()
        renderer   = build_renderer(args.data, cfg_hi.img_hw)
        pixel_mask = load_pixel_mask(args.data, renderer, cfg_hi.img_hw,
                                     frame_mask=cfg_hi.frame_mask).to(device)
        loader = dm.train_dataloader(); it = [iter(loader)]
        def _next():
            try: return next(it[0])
            except StopIteration:
                it[0] = iter(loader); return next(it[0])
        def pair(bs):
            _, pb = _next(); npred = pb.shape[1] - 1
            t = int(torch.randint(1, npred + 1, (1,)))
            return pb[:, 0].to(device), pb[:, t].to(device)
        def window(bs):
            cb, pb = _next()
            return ([cb[:, t].to(device) for t in range(cb.shape[1])],
                    [pb[:, t].to(device) for t in range(pb.shape[1])])
        def rollout_data():
            cb, pb = _next()
            ctx = [cb[:, t].to(device) for t in range(cb.shape[1])]
            gt  = [pb[:, t].to(device) for t in range(1, pb.shape[1])]
            return ctx, pb[:, 0].to(device), gt[:args.roll_steps]
        sources = dict(pixel_mask=pixel_mask, pair=pair, window=window, rollout=rollout_data)
    else:
        pm = None
        def pair(bs):   return synth_frame(cfg_hi, bs, device), synth_frame(cfg_hi, bs, device)
        def window(bs): return ([synth_frame(cfg_hi, bs, device) for _ in range(cfg_hi.n_context_frames)],
                                [synth_frame(cfg_hi, bs, device) for _ in range(cfg_hi.horizon_max + 1)])
        def rollout_data():
            return ([synth_frame(cfg_hi, B, device) for _ in range(cfg_hi.n_context_frames)],
                    synth_frame(cfg_hi, B, device), None)
        sources = dict(pixel_mask=pm, pair=pair, window=window, rollout=rollout_data)
        print('(synthetic data — quality numbers omitted; params/throughput/inference-compute are real)')

    r_no = run_variant('NO hierarchy', cfg_no, device, args, sources)
    r_hi = run_variant('hierarchy',    cfg_hi, device, args, sources)

    # --- verdict table ---
    def pct(a, b): return f'{(b - a) / a * 100:+.0f}%'
    print('\n============ SUMMARY (hierarchy vs no-hierarchy) ============')
    print(f'  AE params        {r_no["ae"]["params"]:6.1f}M -> {r_hi["ae"]["params"]:6.1f}M   ({pct(r_no["ae"]["params"], r_hi["ae"]["params"])})')
    print(f'  dyn params       {r_no["dyn"]["params"]:6.1f}M -> {r_hi["dyn"]["params"]:6.1f}M   ({pct(r_no["dyn"]["params"], r_hi["dyn"]["params"])})')
    print(f'  AE train samp/s  {r_no["ae"]["samp_s"]:6.0f}  -> {r_hi["ae"]["samp_s"]:6.0f}   ({pct(r_no["ae"]["samp_s"], r_hi["ae"]["samp_s"])})')
    print(f'  dyn train samp/s {r_no["dyn"]["samp_s"]:6.0f}  -> {r_hi["dyn"]["samp_s"]:6.0f}   ({pct(r_no["dyn"]["samp_s"], r_hi["dyn"]["samp_s"])})')
    K = cfg_hi.n_slots
    full_no = r_no['roll'][K]['ms_step']
    print(f'\n  rollout ms/step @ full (N={K}):  no-hier {full_no:.1f}  |  hier {r_hi["roll"][K]["ms_step"]:.1f}')
    print(f'  hierarchy variable-width dial (the benefit):')
    for N in r_hi['widths']:
        r = r_hi['roll'][N]
        save = f'{(1 - r["ms_step"]/r_hi["roll"][K]["ms_step"])*100:.0f}% cheaper' if N != K else 'baseline'
        mae = f'  MAE={r["mae"]:.4f}' if r['mae'] is not None else ''
        print(f'    N={N:>4}: {r["ms_step"]:.1f} ms/step  ({save}){mae}')
    print('\n  (params/train-throughput = the cost;  width dial = the benefit)')


if __name__ == '__main__':
    main()

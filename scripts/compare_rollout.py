"""
Head-to-head rollout comparison: end-to-end HFM1D vs the two-phase latent model.

Both models are rolled out from the *same* seed frames / context on the same
validation sequences, and per-step masked MAE (accuracy vs horizon) + per-rollout
wall-clock (compute) are reported.  This decides "does the two-phase design help"
empirically — at matched horizon, and you control the training budget separately.

Usage:
    python scripts/compare_rollout.py \\
        --hfm checkpoints/train_epoch049.pt \\
        --dyn checkpoints_dyn/dyn_epoch049.pt \\
        --data-dir ../data/test --stats-dir ../data/fvm_gen_datasets \\
        --n-predict 10 --n-seqs 32 --reencode-every 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import HFM1D, ContextEncoder, LatentDynamicsTrainer
from dfm.data import build_renderer, FVMSequenceDataset, load_pixel_mask
from infer import get_device, load_stats, find_sim_dirs


def _masked_mae(pred, target, mask):
    m = mask.expand_as(target).bool()
    return (pred[m] - target[m]).abs().mean().item()


def main():
    ap = argparse.ArgumentParser(description='HFM1D vs two-phase rollout comparison')
    ap.add_argument('--hfm', required=True, help='end-to-end HFM1D checkpoint (train_*.pt)')
    ap.add_argument('--dyn', required=True, help='two-phase dynamics checkpoint (dyn_*.pt)')
    ap.add_argument('--data-dir',  required=True)
    ap.add_argument('--stats-dir', default=None)
    ap.add_argument('--n-predict', type=int, default=10)
    ap.add_argument('--n-seqs',    type=int, default=32, help='sequences to average over')
    ap.add_argument('--reencode-every', type=int, default=None,
                    help='override re-anchor cadence for BOTH models (else each uses its cfg)')
    ap.add_argument('--first-frame', type=int, default=20)
    ap.add_argument('--out', default='compare', help='output prefix (.csv / .png)')
    args = ap.parse_args()

    device = get_device()
    print(f'Device: {device}')

    # ---- load end-to-end HFM1D ----
    ckA = torch.load(args.hfm, map_location='cpu', weights_only=False)
    cfgA = ckA['cfg']
    modelA = HFM1D(cfgA).to(device).eval()
    modelA.load_state_dict(ckA['model'], strict=False)
    ceA = ContextEncoder(cfgA).to(device).eval()
    ceA.load_state_dict(ckA['context_encoder'], strict=False)

    # ---- load two-phase dynamics (bundles frozen AE) ----
    ckB = torch.load(args.dyn, map_location='cpu', weights_only=False)
    cfgB = ckB['cfg']
    trB = LatentDynamicsTrainer(cfgB).to(device)
    trB.load(args.dyn)

    n_context = cfgA.n_context_frames
    img       = cfgA.img_size
    reA = args.reencode_every if args.reencode_every is not None else cfgA.reencode_every
    reB = args.reencode_every if args.reencode_every is not None else cfgB.reencode_every
    print(f'HFM1D:      step={ckA.get("global_step","?")}  reencode_every={reA}')
    print(f'two-phase:  step={ckB.get("global_step","?")}  reencode_every={reB}  '
          f'evolve_state={cfgB.evolve_state}')

    # ---- data ----
    stats_dir = Path(args.stats_dir) if args.stats_dir else Path(args.data_dir)
    mean, std = load_stats(stats_dir)
    renderer = build_renderer(Path(args.data_dir), (img, img), device='cpu')
    pm = load_pixel_mask(Path(args.data_dir), renderer, (img, img)).to(device)
    sim_dirs = find_sim_dirs(Path(args.data_dir))
    if not sim_dirs:
        raise RuntimeError(f'No simulation subdirectories found in {args.data_dir}')

    pred_len = args.n_predict + 1
    per_run = max(1, args.n_seqs // len(sim_dirs) + 1)
    seqs = []
    for sim_dir in sim_dirs:
        ds = FVMSequenceDataset.with_cache(sim_dir, renderer, n_context, pred_len, mean, std,
                                           first_frame=args.first_frame, random_context=False)
        if len(ds) == 0:
            continue
        for start in np.linspace(0, len(ds) - 1, num=per_run).astype(int):
            seqs.append((ds, int(start)))
            if len(seqs) >= args.n_seqs:
                break
        if len(seqs) >= args.n_seqs:
            break
    print(f'Evaluating {len(seqs)} sequences (n_predict={args.n_predict})\n')

    maeA = np.zeros(args.n_predict)
    maeB = np.zeros(args.n_predict)
    tA = tB = 0.0
    for ds, start in seqs:
        ctx_seq, pred_seq = ds[start]
        context_frames = [ctx_seq[t:t+1].to(device) * pm for t in range(n_context)]
        pred_gt        = [pred_seq[t:t+1].to(device) * pm for t in range(pred_len)]
        x0 = pred_gt[0]
        with torch.no_grad():
            t0 = time.perf_counter()
            ctx = ceA(context_frames, pixel_mask=pm)
            predsA = modelA(x0, ctx, horizon=args.n_predict, reencode_every=reA, pixel_mask=pm)
            if device.type == 'cuda': torch.cuda.synchronize()
            tA += time.perf_counter() - t0

            t0 = time.perf_counter()
            predsB = trB.rollout(context_frames, x0, n_steps=args.n_predict,
                                 reencode_every=reB, pixel_mask=pm)
            if device.type == 'cuda': torch.cuda.synchronize()
            tB += time.perf_counter() - t0

        for t in range(args.n_predict):
            tgt = pred_gt[1 + t]
            maeA[t] += _masked_mae(predsA[t].float() * pm, tgt, pm)
            maeB[t] += _masked_mae(predsB[t].float(),      tgt, pm)

    n = len(seqs)
    maeA /= n; maeB /= n

    # ---- report ----
    print(f'{"step":>4} {"HFM1D":>10} {"two-phase":>10} {"Δ(dyn-hfm)":>12}')
    for t in range(args.n_predict):
        print(f'{t+1:>4} {maeA[t]:>10.5f} {maeB[t]:>10.5f} {maeB[t]-maeA[t]:>+12.5f}')
    print(f'\nmean MAE     HFM1D={maeA.mean():.5f}   two-phase={maeB.mean():.5f}')
    print(f'rollout time HFM1D={tA/n*1e3:.1f} ms/seq   two-phase={tB/n*1e3:.1f} ms/seq   '
          f'(speedup ×{tA/max(tB,1e-9):.2f})')

    out = Path(args.out)
    import csv
    with open(out.with_suffix('.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['step', 'mae_hfm', 'mae_two_phase'])
        for t in range(args.n_predict):
            wr.writerow([t + 1, f'{maeA[t]:.6f}', f'{maeB[t]:.6f}'])
    print(f'csv → {out.with_suffix(".csv")}')

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs = np.arange(1, args.n_predict + 1)
        plt.figure(figsize=(7, 4))
        plt.plot(xs, maeA, 'o-', label=f'HFM1D (end-to-end)  {tA/n*1e3:.0f} ms')
        plt.plot(xs, maeB, 's-', label=f'two-phase (latent)  {tB/n*1e3:.0f} ms')
        plt.xlabel('rollout step'); plt.ylabel('masked MAE')
        plt.title(f'Rollout accuracy vs horizon  (reencode_every={reA}/{reB})')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out.with_suffix('.png'), dpi=120)
        print(f'plot → {out.with_suffix(".png")}')
    except ImportError:
        print('(matplotlib not available — skipped plot)')


if __name__ == '__main__':
    main()

"""
Inference for the two-phase latent model (frozen AE + latent dynamics).

Loads a phase-2 dynamics checkpoint (which bundles the frozen AE), feeds
n_context ground-truth frames into the ContextEncoder, then rolls the latent
forward from a seed frame and decodes each step.  Re-anchoring cadence is
controlled by --reencode-every (0 = never; pure latent rollout).

Outputs mirror infer.py:
  - frames_gt.npy / frames_pred.npy (+ *_phys.npy)
  - viewer/<run>/t_*.npz
  - images/t{NNN}_ch{C}.png

Usage:
    python infer_dynamics.py --checkpoint checkpoints_dyn/dyn_epoch049.pt \\
                             --data-dir  ../data/test \\
                             --stats-dir ../data/fvm_gen_datasets \\
                             --out-dir   out/infer_dyn
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dfm import DFMConfig, LatentDynamicsTrainer
from dfm.data import build_renderer, FVMSequenceDataset, load_pixel_mask
from dfm.inference_utils import (get_device, load_stats, denorm, find_sim_dirs,
                                 save_images, save_viewer_frames)


def main():
    p = argparse.ArgumentParser(description='Two-phase latent model inference')
    p.add_argument('--checkpoint',    required=True, help='phase-2 dynamics checkpoint (bundles the AE)')
    p.add_argument('--data-dir',      required=True)
    p.add_argument('--stats-dir',     default=None, help='dir with TRAINING stats (defaults to --data-dir)')
    p.add_argument('--out-dir',       required=True)
    p.add_argument('--n-context',     type=int, default=None)
    p.add_argument('--n-predict',     type=int, default=10)
    p.add_argument('--reencode-every', type=int, default=None,
                   help='decode-based re-anchor cadence (0 = pure latent rollout; defaults to cfg)')
    p.add_argument('--seq-start',     type=int, default=None)
    p.add_argument('--first-frame',   type=int, default=20)
    p.add_argument('--no-images',     action='store_true')
    args = p.parse_args()

    device   = get_device()
    out_dir  = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device:  {device}\nOut dir: {out_dir}')

    # ---- load checkpoint (dynamics + context + frozen AE) ----
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg: DFMConfig = ckpt['cfg']
    trainer = LatentDynamicsTrainer(cfg).to(device)
    trainer.load(args.checkpoint)
    n_context = args.n_context if args.n_context is not None else cfg.n_context_frames
    reencode  = args.reencode_every if args.reencode_every is not None else cfg.reencode_every
    print(f'  step={ckpt.get("global_step", "?")}  evolve_state={cfg.evolve_state}  '
          f'reencode_every={reencode}')

    # ---- data ----
    stats_dir = Path(args.stats_dir) if args.stats_dir else data_dir
    mean, std = load_stats(stats_dir)
    renderer   = build_renderer(data_dir, (cfg.img_size, cfg.img_size), device='cpu')
    pixel_mask = load_pixel_mask(data_dir, renderer, (cfg.img_size, cfg.img_size)).to(device)
    sim_dirs = find_sim_dirs(data_dir)
    if not sim_dirs:
        raise RuntimeError(f'No simulation subdirectories found in {data_dir}')
    print(f'  Found {len(sim_dirs)} simulation directories\n')

    pred_len = args.n_predict + 1
    for sim_idx, sim_dir in enumerate(sim_dirs):
        print(f'[{sim_idx+1}/{len(sim_dirs)}] {sim_dir.name}')
        run_out = out_dir / sim_dir.name
        run_out.mkdir(parents=True, exist_ok=True)

        # random_context=False → deterministic context from the run start
        ds = FVMSequenceDataset.with_cache(
            sim_dir, renderer, n_context, pred_len, mean, std,
            first_frame=args.first_frame, random_context=False,
        )
        if len(ds) == 0:
            print('  [skip] no sequences available'); continue

        start = args.seq_start if args.seq_start is not None else len(ds) // 2
        ctx_seq, pred_seq = ds[start]
        context_frames = [ctx_seq[t:t+1].to(device) * pixel_mask for t in range(n_context)]
        pred_gt        = [pred_seq[t:t+1].to(device) * pixel_mask for t in range(pred_len)]
        pred_ts        = [float(ds.paths[start + t].stem[2:]) for t in range(pred_len)]
        x0 = pred_gt[0]

        # ---- latent rollout ----
        preds_list = trainer.rollout(context_frames, x0, n_steps=args.n_predict,
                                     reencode_every=reencode, pixel_mask=pixel_mask)

        preds, gt = [], []
        for t, pred in enumerate(preds_list):
            preds.append(pred.float().cpu())
            gt.append(pred_gt[1 + t].cpu())
            mae = (pred_gt[1 + t] - pred).abs().mean().item()
            print(f'  t={t+1:3d}  MAE={mae:.5f}')

        # ---- save ----
        gt_arr    = torch.cat(gt,    dim=0).numpy()
        pred_arr  = torch.cat(preds, dim=0).numpy()
        gt_phys   = denorm(gt_arr,   mean, std)
        pred_phys = denorm(pred_arr, mean, std) * pixel_mask.cpu().numpy()
        np.save(run_out / 'frames_gt.npy',        gt_arr)
        np.save(run_out / 'frames_pred.npy',      pred_arr)
        np.save(run_out / 'frames_gt_phys.npy',   gt_phys)
        np.save(run_out / 'frames_pred_phys.npy', pred_phys)

        seed_phys = denorm(pred_gt[0].cpu().numpy(), mean, std) * pixel_mask.cpu().numpy()
        save_viewer_frames(seed_phys, pred_phys, pred_ts, n_context=1,
                           run_name=sim_dir.name, viewer_dir=out_dir / 'viewer')
        if not args.no_images:
            save_images(gt_arr, pred_arr, run_out, mean, std)
        print(f'  Saved → {run_out}')

    print(f'\nDone.  Viewer output → {out_dir / "viewer"}')


if __name__ == '__main__':
    main()

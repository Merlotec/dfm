"""
Inference for the two-phase latent model (frozen AE + latent dynamics).

Loads a phase-2 dynamics checkpoint (which bundles the frozen AE), encodes the
ground-truth zero-motion seed L_0, then rolls the latent forward and decodes each step.

Outputs mirror infer.py:
  - frames_gt.npy / frames_pred.npy (+ *_phys.npy)
  - viewer/<run>/t_*.npz
  - images/t{NNN}_ch{C}.png

Usage:
    python infer_dynamics.py --checkpoint checkpoints_dyn/dyn_epoch049.pt \
                             --data-dir  ../data/test \
                             --stats-dir ../data/fvm_gen_datasets \
                             --out-dir   out/infer_dyn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dfm import DFMConfig, RolloutTrainer
from dfm.data import build_renderer, FVMSequenceDataset, load_pixel_mask

CHANNEL_NAMES = ['rho', 'u', 'v', 'p']

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def load_stats(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    for p in [data_dir / 'hfm_input_stats.json']:
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            return torch.tensor(s['mean']), torch.tensor(s['std'])
    raise FileNotFoundError('No normalisation stats found.')

def denorm(frames: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] normalised → physical units."""
    m = mean.numpy()[None, :, None, None]
    s = std.numpy()[None, :, None, None]
    return frames * s + m

def save_viewer_frames(
    gt_phys: np.ndarray,
    pred_phys: np.ndarray,
    timestamps: list[float],
    n_context: int,
    run_name: str,
    viewer_dir: Path,
):
    """Write t_*.npz files in the format expected by fvm_viewer/viewer.py -c."""
    run_dir = viewer_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    T = len(timestamps)
    for i, ts in enumerate(timestamps):
        if i < n_context:
            grid    = gt_phys[i].astype(np.float32)
            is_seed = True
        else:
            grid    = pred_phys[i - n_context].astype(np.float32)
            is_seed = False
        np.savez(run_dir / f't_{ts:.4g}.npz',
                 grid=grid, t=np.float32(ts), is_seed=np.bool_(is_seed))

    print(f'  Saved {T} viewer frames → {run_dir}')

def save_images(gt: np.ndarray, pred: np.ndarray, out_dir: Path,
                mean: torch.Tensor, std: torch.Tensor):
    """Save side-by-side GT vs prediction PNGs, one per timestep per channel."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available — skipping PNG output')
        return

    img_dir = out_dir / 'images'
    img_dir.mkdir(exist_ok=True)

    gt_phys   = denorm(gt,   mean, std)
    pred_phys = denorm(pred, mean, std)

    T, C = gt.shape[:2]
    for t in range(T):
        for c in range(C):
            _, axes = plt.subplots(1, 3, figsize=(14, 4))
            vmin = gt_phys[t, c].min()
            vmax = gt_phys[t, c].max()

            axes[0].imshow(gt_phys[t, c],   vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[1].imshow(pred_phys[t, c], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            err = np.abs(gt_phys[t, c] - pred_phys[t, c])
            axes[2].imshow(err, cmap='hot')

            axes[0].set_title(f'Ground truth — {CHANNEL_NAMES[c]}')
            axes[1].set_title(f'Prediction   — {CHANNEL_NAMES[c]}')
            axes[2].set_title(f'|Error| max={err.max():.3f}')
            for ax in axes:
                ax.axis('off')

            plt.suptitle(f't={t}  channel={CHANNEL_NAMES[c]}', fontsize=12)
            plt.tight_layout()
            plt.savefig(img_dir / f't{t:03d}_ch{c}_{CHANNEL_NAMES[c]}.png',
                        dpi=100, bbox_inches='tight')
            plt.close()

    print(f'  Saved {T * C} images → {img_dir}')

def find_sim_dirs(data_dir: Path) -> list[Path]:
    sim_dirs = []
    if (data_dir / 'shared_mesh.pkl').exists():
        sim_dirs.extend([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('run')])
    else:
        for p in data_dir.iterdir():
            if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                sim_dirs.extend([s for s in p.iterdir() if s.is_dir() and s.name.startswith('run')])
    return sorted(sim_dirs)


def main():
    p = argparse.ArgumentParser(description='Two-phase latent model inference')
    p.add_argument('--checkpoint',    required=True, help='phase-2 dynamics checkpoint (bundles the AE)')
    p.add_argument('--data-dir',      required=True)
    p.add_argument('--stats-dir',     default=None, help='dir with TRAINING stats (defaults to --data-dir)')
    p.add_argument('--out-dir',       required=True)
    p.add_argument('--n-predict',     type=int, default=10)
    p.add_argument('--seq-start',     type=int, default=None)
    p.add_argument('--first-frame',   type=int, default=20)
    p.add_argument('--no-images',     action='store_true')
    args = p.parse_args()

    device   = get_device()
    out_dir  = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device:  {device}\nOut dir: {out_dir}')

    # ---- load checkpoint (dynamics + frozen AE) ----
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg: DFMConfig = ckpt['cfg']
    trainer = RolloutTrainer(cfg).to(device)
    trainer.load(args.checkpoint)
    print(f'  step={ckpt.get("global_step", "?")}')

    # ---- data ----
    stats_dir = Path(args.stats_dir) if args.stats_dir else data_dir
    mean, std = load_stats(stats_dir)
    renderer   = build_renderer(data_dir, cfg.img_hw, device='cpu')
    pixel_mask = load_pixel_mask(data_dir, renderer, cfg.img_hw, frame_mask=cfg.frame_mask).to(device)
    sim_dirs = find_sim_dirs(data_dir)
    if not sim_dirs:
        raise RuntimeError(f'No simulation subdirectories found in {data_dir}')
    print(f'  Found {len(sim_dirs)} simulation directories\n')

    pred_len = args.n_predict + 1
    for sim_idx, sim_dir in enumerate(sim_dirs):
        print(f'[{sim_idx+1}/{len(sim_dirs)}] {sim_dir.name}')
        run_out = out_dir / sim_dir.name
        run_out.mkdir(parents=True, exist_ok=True)

        ds = FVMSequenceDataset.with_cache(
            sim_dir, renderer, cfg.n_context_frames, pred_len, mean, std,
            first_frame=args.first_frame, random_context=False,
        )
        if len(ds) == 0:
            print('  [skip] no sequences available'); continue

        start = args.seq_start if args.seq_start is not None else len(ds) // 2
        _, pred_seq = ds[start]
        valid = pixel_mask[:, :1]
        pred_gt = [pred_seq[t:t+1].to(device) * valid for t in range(pred_len)]
        pred_ts = [float(ds.paths[start + t].stem[2:]) for t in range(pred_len)]
        x0 = pred_gt[0]

        # ---- latent rollout ----
        preds_list = trainer.rollout(x0, n_steps=args.n_predict, pixel_mask=pixel_mask)

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
        pred_phys = denorm(pred_arr, mean, std) * pixel_mask[:, :1].cpu().numpy()
        np.save(run_out / 'frames_gt.npy',        gt_arr)
        np.save(run_out / 'frames_pred.npy',      pred_arr)
        np.save(run_out / 'frames_gt_phys.npy',   gt_phys)
        np.save(run_out / 'frames_pred_phys.npy', pred_phys)

        seed_phys = denorm(pred_gt[0].cpu().numpy(), mean, std) * pixel_mask[:, :1].cpu().numpy()
        save_viewer_frames(seed_phys, pred_phys, pred_ts, n_context=1,
                           run_name=sim_dir.name, viewer_dir=out_dir / 'viewer')
        if not args.no_images:
            save_images(gt_arr, pred_arr, run_out, mean, std)
        print(f'  Saved → {run_out}')

    print(f'\nDone.  Viewer output → {out_dir / "viewer"}')


if __name__ == '__main__':
    main()

"""
DFM inference script.

Loads a RolloutGANTrainer checkpoint, feeds n_context ground-truth frames into
the ContextEncoder, then rolls the seed frame forward in latent space and
decodes each step.  Unlike a pixel-autoregressive model, DFM encodes the seed
once and advances a compact latent state; intermediate frames are decoded from
that latent (with periodic re-encoding controlled by cfg.reencode_every).

Outputs are saved to --out-dir as:
  - frames_gt.npy / frames_pred.npy            — normalised arrays [T, C, H, W]
  - frames_gt_phys.npy / frames_pred_phys.npy  — denormalised physical values
  - viewer/<run>/t_*.npz                        — fvm_viewer-compatible frames
  - images/t{NNN}_ch{C}.png                     — GT vs prediction per channel

Usage:
    python infer.py --checkpoint checkpoints/train_epoch009.pt \\
                    --data-dir   /path/to/sim_dataset \\
                    --out-dir    out/infer
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dfm import HFM1D, ContextEncoder, HFMDiscriminator
from dfm.config import HFM1DConfig
from dfm.data import build_renderer, FVMSequenceDataset, load_pixel_mask

# Rendered primitive channels: [v_x, v_y, rho, T]  (see solver state_to_primative)
CHANNEL_NAMES = ['vx', 'vy', 'rho', 'T']

STATS_FILES = ['hfm_input_stats.json', 'dfm_input_stats.json']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_stats(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    for name in STATS_FILES:
        p = data_dir / name
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            return torch.tensor(s['mean']), torch.tensor(s['std'])
    raise FileNotFoundError(f'No normalisation stats found in {data_dir} '
                            f'(looked for {STATS_FILES}).')


def find_sim_dirs(data_dir: Path) -> list[Path]:
    """Subdirectories containing frame files (t_*.npz) — robust to run naming."""
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith('.')
        and any(f.name.startswith('t_') and f.name.endswith('.npz') for f in p.iterdir())
    )


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

    for i, ts in enumerate(timestamps):
        if i < n_context:
            grid, is_seed = gt_phys[i].astype(np.float32), True
        else:
            grid, is_seed = pred_phys[i - n_context].astype(np.float32), False
        np.savez(run_dir / f't_{ts:.4g}.npz',
                 grid=grid, t=np.float32(ts), is_seed=np.bool_(is_seed))

    print(f'  Saved {len(timestamps)} viewer frames → {run_dir}')


def save_images(gt: np.ndarray, pred: np.ndarray, out_dir: Path,
                mean: torch.Tensor, std: torch.Tensor):
    """Save GT | prediction | |error| PNGs, one per timestep per channel."""
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
            vmin, vmax = gt_phys[t, c].min(), gt_phys[t, c].max()

            axes[0].imshow(gt_phys[t, c],   vmin=vmin, vmax=vmax, cmap='RdBu_r')
            axes[1].imshow(pred_phys[t, c], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            err = np.abs(gt_phys[t, c] - pred_phys[t, c])
            axes[2].imshow(err, cmap='hot')

            axes[0].set_title(f'Ground truth — {CHANNEL_NAMES[c]}')
            axes[1].set_title(f'Prediction   — {CHANNEL_NAMES[c]}')
            axes[2].set_title(f'|Error|  max={err.max():.3f}')
            for ax in axes:
                ax.axis('off')

            plt.suptitle(f't={t}  channel={CHANNEL_NAMES[c]}', fontsize=12)
            plt.tight_layout()
            plt.savefig(img_dir / f't{t:03d}_ch{c}_{CHANNEL_NAMES[c]}.png',
                        dpi=100, bbox_inches='tight')
            plt.close()

    print(f'  Saved {T * C} images → {img_dir}')


# ---------------------------------------------------------------------------
# Discriminator saliency
# ---------------------------------------------------------------------------

def disc_saliency(discriminator: HFMDiscriminator, frame: torch.Tensor,
                  x_prev: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Gradient of the discriminator logit w.r.t. input pixels → [1, 1, H, W]."""
    inp = frame.detach().float().requires_grad_(True)
    logit = discriminator(inp, x_prev.detach(), context.detach())
    logit.mean().backward()
    assert inp.grad is not None
    return inp.grad.abs().mean(dim=1, keepdim=True)


def save_disc_saliency(saliency_fake, saliency_real, pred_phys, gt_phys, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available — skipping saliency output')
        return

    sal_dir = out_dir / 'disc_saliency'
    sal_dir.mkdir(exist_ok=True)

    def _norm(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    for t, (sf, sr) in enumerate(zip(saliency_fake, saliency_real)):
        _, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(pred_phys[t, 0], cmap='RdBu_r')
        axes[0].set_title(f'Prediction (vx)  t={t}')
        axes[1].imshow(gt_phys[t, 0], cmap='RdBu_r')
        axes[1].set_title(f'Ground truth (vx)  t={t}')
        axes[2].imshow(_norm(sf.squeeze().cpu().numpy()), cmap='hot')
        axes[2].set_title('Disc saliency — fake')
        axes[3].imshow(_norm(sr.squeeze().cpu().numpy()), cmap='hot')
        axes[3].set_title('Disc saliency — real')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(sal_dir / f't{t:03d}_disc_saliency.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f'  Saved {len(saliency_fake)} saliency maps → {sal_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DFM inference')
    parser.add_argument('--checkpoint',   required=True, help='Path to .pt checkpoint')
    parser.add_argument('--data-dir',     required=True, help='Dataset directory (subdirs of t_*.npz)')
    parser.add_argument('--stats-dir',    default=None,
                        help='Directory holding the TRAINING normalisation stats '
                             '(defaults to --data-dir). Use this when running on a '
                             'test set that has no stats of its own.')
    parser.add_argument('--out-dir',      required=True, help='Output directory')
    parser.add_argument('--n-context',    type=int, default=None,
                        help='Context frames (defaults to cfg.n_context_frames)')
    parser.add_argument('--n-predict',    type=int, default=10,
                        help='Number of frames to roll out and decode')
    parser.add_argument('--reencode-every', type=int, default=None,
                        help='Override cfg.reencode_every for this rollout (0 = never)')
    parser.add_argument('--seq-start',    type=int, default=None,
                        help='Frame index to start from (defaults to middle of run)')
    parser.add_argument('--first-frame',  type=int, default=20,
                        help='Skip this many initial transient frames')
    parser.add_argument('--no-images',    action='store_true', help='Skip PNG generation')
    parser.add_argument('--disc-saliency', action='store_true',
                        help='Compute and save discriminator saliency heatmaps')
    args = parser.parse_args()

    device   = get_device()
    out_dir  = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device:  {device}')
    print(f'Out dir: {out_dir}')

    # ---- load checkpoint ----
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg: HFM1DConfig = ckpt['cfg']
    if args.reencode_every is not None:
        cfg.reencode_every = args.reencode_every
    print(f'  step={ckpt.get("global_step", "?")}  '
          f'n_slots={cfg.n_slots}  integrator={cfg.integrator}  '
          f'reencode_every={cfg.reencode_every}')

    model = HFM1D(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if unexpected:
        print(f'  [warn] model unexpected keys: {unexpected}')
    if missing:
        print(f'  New/missing model keys (random init): {missing}')
    model.eval()

    context_encoder = ContextEncoder(cfg).to(device)
    if 'context_encoder' in ckpt:
        context_encoder.load_state_dict(ckpt['context_encoder'])
    else:
        print('  [warn] no context_encoder in checkpoint; using random weights')
    context_encoder.eval()

    discriminator: HFMDiscriminator | None = None
    if args.disc_saliency and 'discriminator' in ckpt:
        discriminator = HFMDiscriminator(cfg).to(device)
        discriminator.load_state_dict(ckpt['discriminator'], strict=False)
        discriminator.eval()
        print('  Discriminator loaded for saliency.')
    elif args.disc_saliency:
        print('  [warn] no discriminator in checkpoint — skipping saliency')

    n_context = args.n_context if args.n_context is not None else cfg.n_context_frames

    # ---- load data ----
    print(f'\nLoading data from {data_dir}')
    stats_dir = Path(args.stats_dir) if args.stats_dir else data_dir
    mean, std = load_stats(stats_dir)   # must be the TRAINING stats, not test-set stats
    if stats_dir != data_dir:
        print(f'  Using training stats from {stats_dir}')
    renderer   = build_renderer(data_dir, (cfg.img_size, cfg.img_size), device='cpu')
    pixel_mask = load_pixel_mask(data_dir, renderer, (cfg.img_size, cfg.img_size)).to(device)

    sim_dirs = find_sim_dirs(data_dir)
    if not sim_dirs:
        raise RuntimeError(f'No simulation subdirectories found in {data_dir}')
    print(f'  Found {len(sim_dirs)} simulation directories\n')

    # context frames + seed + n_predict targets (seed is targets[-1]'s predecessor)
    seq_len = n_context + args.n_predict + 1

    for sim_idx, sim_dir in enumerate(sim_dirs):
        print(f'[{sim_idx+1}/{len(sim_dirs)}] {sim_dir.name}')
        run_out = out_dir / sim_dir.name
        run_out.mkdir(parents=True, exist_ok=True)

        ds = FVMSequenceDataset.with_cache(
            sim_dir, renderer, seq_len, mean, std, first_frame=args.first_frame
        )
        if len(ds) == 0:
            print(f'  [skip] no sequences available')
            continue

        start = args.seq_start if args.seq_start is not None else len(ds) // 2
        seq   = ds[start]
        frames_gt  = [seq[t:t+1].to(device) * pixel_mask for t in range(seq_len)]
        timestamps = [float(ds.paths[start + t].stem[2:]) for t in range(seq_len)]

        x0 = frames_gt[n_context]

        # ---- encode context once, then roll the latent forward ----
        with torch.no_grad():
            context = context_encoder(frames_gt[:n_context], pixel_mask=pixel_mask)
            preds_list = model(x0, context, horizon=args.n_predict, pixel_mask=pixel_mask)

        preds, gt = [], []
        for t, pred in enumerate(preds_list):
            pred = pred.float() * pixel_mask
            preds.append(pred.cpu())
            target = frames_gt[n_context + 1 + t]
            gt.append(target.cpu())
            mae = (target - pred).abs().mean().item()
            print(f'  t={t+1:3d}  MAE={mae:.5f}')

        # ---- optional discriminator saliency ----
        saliency_fakes, saliency_reals = [], []
        if discriminator is not None:
            for t, pred in enumerate(preds_list):
                x_prev = frames_gt[n_context + t]           # true previous frame
                target = frames_gt[n_context + 1 + t]
                pred_m = pred.float() * pixel_mask
                saliency_fakes.append(disc_saliency(discriminator, pred_m, x_prev, context))
                saliency_reals.append(disc_saliency(discriminator, target, x_prev, context))

        # ---- save outputs ----
        gt_arr    = torch.cat(gt,    dim=0).numpy()
        pred_arr  = torch.cat(preds, dim=0).numpy()
        gt_phys   = denorm(gt_arr,   mean, std)
        pred_phys = denorm(pred_arr, mean, std) * pixel_mask.cpu().numpy()

        np.save(run_out / 'frames_gt.npy',        gt_arr)
        np.save(run_out / 'frames_pred.npy',      pred_arr)
        np.save(run_out / 'frames_gt_phys.npy',   gt_phys)
        np.save(run_out / 'frames_pred_phys.npy', pred_phys)

        context_phys = denorm(
            torch.cat([frames_gt[t].cpu() for t in range(n_context)], dim=0).numpy(),
            mean, std,
        ) * pixel_mask.cpu().numpy()
        all_ts = timestamps[:n_context] + timestamps[n_context + 1: n_context + 1 + args.n_predict]
        save_viewer_frames(
            gt_phys    = context_phys,
            pred_phys  = pred_phys,
            timestamps = all_ts,
            n_context  = n_context,
            run_name   = sim_dir.name,
            viewer_dir = out_dir / 'viewer',
        )

        if not args.no_images:
            save_images(gt_arr, pred_arr, run_out, mean, std)

        if discriminator is not None and saliency_fakes:
            save_disc_saliency(saliency_fakes, saliency_reals, pred_phys, gt_phys, run_out)

        print(f'  Saved → {run_out}')

    print(f'\nDone.  Viewer output → {out_dir / "viewer"}')


if __name__ == '__main__':
    main()

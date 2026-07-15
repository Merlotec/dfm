"""Shared inference I/O helpers: device, stats, run discovery, and frame saving."""

import json
from pathlib import Path

import numpy as np
import torch

# Rendered primitive channels: [v_x, v_y, rho, T]  (see solver state_to_primative)
CHANNEL_NAMES = ['vx', 'vy', 'rho', 'T']
STATS_FILES = ['hfm_input_stats.json', 'dfm_input_stats.json']


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


def save_viewer_frames(gt_phys: np.ndarray, pred_phys: np.ndarray,
                       timestamps: list, n_context: int, run_name: str, viewer_dir: Path,
                       shards: list | None = None):
    """Write t_*.npz files in the format expected by fvm_viewer/viewer.py -c.

    ``shards`` (optional) is a per-frame list aligned with ``timestamps``; each entry
    is either None or a dict {'xy': [M,2] float, 'act': [M] float} of shard pixel
    positions and their active(1)→shadow(0) gate, saved as extra npz keys so the
    viewer can show the adaptive-discretisation cloud beside the fluid state."""
    run_dir = viewer_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for i, ts in enumerate(timestamps):
        if i < n_context:
            grid, is_seed = gt_phys[i].astype(np.float32), True
        else:
            grid, is_seed = pred_phys[i - n_context].astype(np.float32), False
        extra = {}
        if shards is not None and shards[i] is not None:
            extra['shard_xy']  = shards[i]['xy'].astype(np.float32)
            extra['shard_act'] = shards[i]['act'].astype(np.float32)
        np.savez(run_dir / f't_{ts:.4g}.npz',
                 grid=grid, t=np.float32(ts), is_seed=np.bool_(is_seed), **extra)
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

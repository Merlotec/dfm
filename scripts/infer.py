"""DFM inference — reconstruct / roll out held-out test sequences from a checkpoint.

Two modes, auto-detected from the checkpoint contents:

  * phase-1 AE checkpoint ('ae' key): TEACHER-FORCED reconstruction.  Each true
    consecutive pair is encoded, its increment composed onto the map, and the
    frame decoded from X_0.  Shows what the transport(+detail) representation can
    capture — the reconstruction quality (the r/b eval), not prediction: without
    the evolution operator you cannot advance the latent without ground truth.

  * phase-2 checkpoint ('evo' key) + --ae AE.pt: AUTONOMOUS rollout.  Seed from
    X_0, roll the evolution operator forward, decode each step.  True prediction,
    no ground truth consumed after X_0.

Per run, under --out-dir/<run>/:
    frames_gt.npy / frames_pred.npy            [T, C, H, W] normalised
    frames_gt_phys.npy / frames_pred_phys.npy  denormalised (physical units)
    images/t{NNN}.png                          GT | pred | |error|, per channel
    metrics.csv                                per-step masked MAE + persistence r/b
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfm import DFMConfig, LatentAutoencoder, EvolutionOperator
from dfm.autoencoder import remap_ae_pyramid_keys, strip_compile_prefix
from dfm.warp import masked_source
from dfm.data import build_renderer, load_pixel_mask, FVMSequenceDataset
from dfm.warp import identity_map
from dfm.distributed import pick_device

_ROOT = Path(__file__).resolve().parents[1]


def load_stats(data_dir: Path):
    """(mean, std) [C] from the dataset's hfm_input_stats.json (same file training used)."""
    with open(data_dir / 'hfm_input_stats.json') as f:
        s = json.load(f)
    return torch.tensor(s['mean']), torch.tensor(s['std'])


@torch.no_grad()
def reconstruct(ae, frames, mask, use_detail):
    """Phase-1 teacher-forced reconstruction. frames [T,C,H,W] (normalised).
    Returns preds [T,C,H,W] (pred[0] == X_0, pred[s] = decode of the true increment
    up to step s)."""
    device = frames.device
    T, C, H, W = frames.shape
    x0 = frames[0:1]
    x0m = masked_source(x0, mask, ae.cfg.warp_fill_holes)
    D, G = identity_map(1, C, H, W, device, torch.float32)
    preds = [x0]
    for s in range(1, T):
        L = ae.encode(frames[s - 1:s], frames[s:s + 1], mask)
        xhat, D, G = ae.decoder.step(L, D, G, x0m, use_detail=use_detail)
        preds.append(xhat)
    return torch.cat(preds, 0)


@torch.no_grad()
def autonomous(ae, evo, frames, mask, use_detail):
    """Phase-2 autonomous rollout from X_0. frames only supplies X_0 (and the GT for
    scoring).  Returns preds [T,C,H,W] with pred[0] == X_0."""
    device = frames.device
    T, C, H, W = frames.shape
    x0 = frames[0:1]
    x0m = masked_source(x0, mask, ae.cfg.warp_fill_holes)
    L = ae.encode(x0, x0, mask)
    D, G = identity_map(1, C, H, W, device, torch.float32)
    preds = [x0]
    for s in range(T - 1):
        L = evo(L, step_idx=s)
        xhat, D, G = ae.decoder.step(L, D, G, x0m, use_detail=use_detail)
        preds.append(xhat)
    return torch.cat(preds, 0)


def write_viewer_frames(viewer_run_dir: Path, pred_phys, timestamps, mask_hw):
    """fvm_viewer comparison format: one t_{ts}.npz per frame, each with
    grid [C,H,W] (denormalised physical field), t, and is_seed (True for X_0 only).
    This is what comp_server.sh's -c dir expects — the viewer pairs these against
    the real run by matching run name + timestamp.

    The collider region is set to 0 (physical) to match GT's renderer fill=0.  The
    model never learns that region (masked out of the loss), and denormalising its
    ~0 normalised output would otherwise show the channel MEAN there, not 0 — a
    spurious bright blob where GT is empty."""
    viewer_run_dir.mkdir(parents=True, exist_ok=True)
    for i, ts in enumerate(timestamps):
        np.savez(viewer_run_dir / f't_{ts:.4g}.npz',
                 grid=(pred_phys[i] * mask_hw).astype(np.float32),
                 t=np.float32(ts), is_seed=np.bool_(i == 0))


def save_images(gt, pred, mask, out_dir: Path):
    """gt/pred [T,C,H,W] numpy (normalised); one PNG per timestep, rows=channels,
    cols = [GT, pred, |error|].  Non-fluid pixels blanked via the mask."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    T, C, H, W = gt.shape
    m = mask[0, 0].astype(bool) if mask is not None else np.ones((H, W), bool)
    for t in range(T):
        fig, ax = plt.subplots(C, 3, figsize=(9, 3 * C))
        ax = ax.reshape(C, 3)
        for c in range(C):
            g = np.where(m, gt[t, c], np.nan)
            p = np.where(m, pred[t, c], np.nan)
            e = np.where(m, np.abs(gt[t, c] - pred[t, c]), np.nan)
            vmin, vmax = np.nanmin(g), np.nanmax(g)
            ax[c, 0].imshow(g, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[c, 1].imshow(p, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[c, 2].imshow(e, cmap='magma')
            for j, ttl in enumerate(('GT', 'pred', '|err|')):
                ax[c, j].set_title(f'ch{c} {ttl}' if c == 0 or True else ttl)
                ax[c, j].axis('off')
        fig.suptitle(f't = {t}')
        fig.tight_layout()
        fig.savefig(out_dir / f't{t:03d}.png', dpi=70)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='DFM inference on held-out sequences')
    p.add_argument('--checkpoint', required=True, help='phase-1 AE ckpt, or phase-2 evo ckpt')
    p.add_argument('--ae', default=None, help='AE ckpt to pair with a phase-2 evo checkpoint')
    p.add_argument('--data', type=Path, default=_ROOT.parent / 'data' / 'test')
    p.add_argument('--out-dir', type=Path, default=_ROOT / 'out' / 'infer')
    p.add_argument('--n-predict', type=int, default=10, help='rollout length')
    p.add_argument('--n-runs', type=int, default=4, help='how many test runs to render')
    p.add_argument('--first-frame', type=int, default=0)
    p.add_argument('--seq-start', type=int, default=None, help='start index (default: run middle)')
    p.add_argument('--no-detail', action='store_true', help='decode transport only (skip DetailHead)')
    p.add_argument('--no-images', action='store_true', help='skip PNGs (metrics + npy only)')
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg: DFMConfig = ckpt['cfg']                          # exact trained architecture
    use_detail = not args.no_detail

    ae = LatentAutoencoder(cfg).to(device).eval()
    evo = None
    if 'evo' in ckpt and 'ae' not in ckpt:               # phase-2 evo checkpoint
        if args.ae is None:
            raise SystemExit('phase-2 checkpoint needs --ae <AE checkpoint>')
        ae_ckpt = torch.load(args.ae, map_location='cpu', weights_only=False)
        _sd = remap_ae_pyramid_keys(strip_compile_prefix(ae_ckpt['ae']))
        _miss, _unexp = ae.load_state_dict(_sd, strict=False)
        if _miss or _unexp:
            print(f'  [warn] AE only partially loaded: {len(_miss)} missing, '
                  f'{len(_unexp)} unexpected keys')
        evo = EvolutionOperator(cfg).to(device).eval()
        evo.load_state_dict(strip_compile_prefix(ckpt['evo']))
        mode = 'phase-2 autonomous rollout'
    else:                                                # phase-1 AE checkpoint
        _sd = remap_ae_pyramid_keys(strip_compile_prefix(ckpt['ae']))
        _miss, _unexp = ae.load_state_dict(_sd, strict=False)
        if _miss or _unexp:
            print(f'  [warn] AE only partially loaded: {len(_miss)} missing, '
                  f'{len(_unexp)} unexpected keys')
        mode = 'phase-1 teacher-forced reconstruction'
    print(f'device={device}  mode={mode}  detail={use_detail}  '
          f'pyramid_levels={cfg.warp_pyramid_levels}')

    mean, std = load_stats(args.data)
    mean_d, std_d = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    # mesh discovery (flat single-mesh dir, or a dir of mesh_*/ subdirs)
    mesh_dirs = ([args.data] if (args.data / 'shared_mesh.pkl').exists()
                 else [d for d in sorted(args.data.iterdir())
                       if d.is_dir() and (d / 'shared_mesh.pkl').exists()])
    if not mesh_dirs:
        raise SystemExit(f'no shared_mesh.pkl in {args.data} or its subdirs')

    runs = []
    for mdir in mesh_dirs:
        renderer = build_renderer(mdir, cfg.img_hw, device='cpu')
        pmask = load_pixel_mask(mdir, renderer, cfg.img_hw, frame_mask=cfg.frame_mask).to(device)
        for sdir in sorted(d for d in mdir.iterdir() if d.is_dir() and d.name.startswith('run')):
            runs.append((sdir, renderer, pmask))
    print(f'found {len(runs)} runs across {len(mesh_dirs)} mesh(es); rendering {min(args.n_runs, len(runs))}\n')

    T = args.n_predict + 1
    all_mae = []
    for i, (sdir, renderer, pmask) in enumerate(runs[:args.n_runs]):
        ds = FVMSequenceDataset.with_cache(sdir, renderer, n_context=1, pred_len=T,
                                           mean=mean, std=std, first_frame=args.first_frame,
                                           random_context=False)
        if len(ds) == 0:
            print(f'[{i+1}] {sdir.name}: too short, skip'); continue
        start = args.seq_start if args.seq_start is not None else len(ds) // 2
        start = min(start, len(ds) - 1)
        _, seq = ds[start]                               # [T, C, H, W]
        frames = seq.to(device)
        # real timestamps for these frames — so the viewer aligns pred to GT
        timestamps = [float(ds.paths[start + t].stem[2:]) for t in range(T)]

        if evo is not None:
            pred = autonomous(ae, evo, frames, pmask, use_detail)
        else:
            pred = reconstruct(ae, frames, pmask, use_detail)

        m = pmask[:, :1].bool()                          # [1,1,H,W]
        # per-step masked MAE, and persistence baseline |X_0 - X_s| for r/b
        run_out = args.out_dir / sdir.name
        run_out.mkdir(parents=True, exist_ok=True)
        rows = [('t', 'mae', 'persist', 'r_over_b')]
        for t in range(1, T):
            err = (pred[t:t+1] - frames[t:t+1]).abs()[m.expand_as(pred[t:t+1])].mean().item()
            base = (frames[0:1] - frames[t:t+1]).abs()[m.expand_as(frames[t:t+1])].mean().item()
            rows.append((t, f'{err:.5f}', f'{base:.5f}', f'{err/max(base,1e-9):.3f}'))
            all_mae.append(err)
        with open(run_out / 'metrics.csv', 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        gt_n = frames.cpu().numpy(); pr_n = pred.cpu().numpy()
        pred_phys = pr_n * std_d.numpy() + mean_d.numpy()
        np.save(run_out / 'frames_gt.npy', gt_n)
        np.save(run_out / 'frames_pred.npy', pr_n)
        np.save(run_out / 'frames_gt_phys.npy', gt_n * std_d.numpy() + mean_d.numpy())
        np.save(run_out / 'frames_pred_phys.npy', pred_phys)
        # fvm_viewer comparison frames (what comp_server.sh opens); collider zeroed
        # to match GT's fill (see write_viewer_frames)
        mask_hw = pmask[0, 0].cpu().numpy()              # [H,W], 1=fluid 0=collider
        write_viewer_frames(args.out_dir / 'viewer' / sdir.name, pred_phys, timestamps, mask_hw)
        if not args.no_images:
            save_images(gt_n, pr_n, pmask.cpu().numpy(), run_out / 'images')
        last = float(rows[-1][3])
        print(f'[{i+1}] {sdir.name}: final-step r/b={last:.3f}  -> {run_out}')

    if all_mae:
        print(f'\nmean masked MAE over all steps/runs: {np.mean(all_mae):.5f}')


if __name__ == '__main__':
    main()

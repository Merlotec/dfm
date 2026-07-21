"""Preflight: catch production-batch failures in ~30s before a long run.

A staged model hides its worst moments at the curriculum boundaries — stage B
turning on (DetailHead memory), the GAN turning on (discriminator + its own
graph).  The smooth part of a run can't reveal them; this does, by FORCING every
regime on from step 0 and running real training steps at the real batch on the
real device.  Synthetic data, so no dataset needed.

Catches: stale hyperparams keys (DFMConfig(**m) throws), NaN losses, shape bugs,
and — the one that keeps biting — the stage-B / GAN activation-memory cliff.

Usage
-----
On Dawn (the real test — one rank, production batch, actual XPU):
    python scripts/preflight.py --batch-size <prod_batch_per_rank>

Locally (just verify the script + code paths run):
    python scripts/preflight.py --batch-size 1 --img-size 64 --steps 1

Exit code 0 = safe to launch; nonzero = it would fail (message says why).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfm import DFMConfig, AutoencoderTrainer
from dfm.distributed import pick_device

HYPERPARAMS = Path(__file__).resolve().parents[1] / 'hyperparams.json'


def load_config(overrides):
    from dataclasses import fields
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    valid = {fld.name for fld in fields(DFMConfig)}
    m = {k: v for k, v in hp['model'].items() if k in valid}
    m.update(overrides)
    # dataclass wants tuples where json gave lists
    for k in ('warp_vel_channels', 'warp_flow_alpha'):
        if k in m and isinstance(m[k], list):
            m[k] = tuple(m[k])
    return DFMConfig(**m), hp.get('training', {})


def peak_gb(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == 'xpu':
        return torch.xpu.max_memory_allocated() / 1e9
    return float('nan')


def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'xpu':
        torch.xpu.reset_peak_memory_stats()


def total_gb(device):
    try:
        if device.type == 'cuda':
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        if device.type == 'xpu':
            return torch.xpu.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, required=True,
                   help='PER-RANK batch — set this to your production value')
    p.add_argument('--steps', type=int, default=2, help='steps per regime')
    p.add_argument('--img-size', type=int, default=None, help='override (local test only)')
    args = p.parse_args()

    device = pick_device()
    overrides = {}
    if args.img_size:
        overrides['img_size'] = args.img_size
    cfg, _ = load_config(overrides)

    print(f'device={device}  total_mem={total_gb(device):.1f}GB')
    print(f'config: img={cfg.img_size} n_slots={cfg.n_slots} detail_slots={cfg.n_detail_slots} '
          f'd_model={cfg.d_model} detail_res={cfg.warp_detail_res} grad_ckpt={cfg.grad_checkpoint}')
    print(f'batch(per-rank)={args.batch_size}  steps/regime={args.steps}\n')

    trainer = AutoencoderTrainer(
        cfg, gan_start_step=10_000, total_steps=100_000,
        norm_stats=(torch.zeros(cfg.in_channels), torch.ones(cfg.in_channels)),
    ).to(device)

    B, K1, C, H, W = args.batch_size, cfg.ae_max_delta + 1, cfg.in_channels, cfg.img_size, cfg.img_size
    mask = torch.ones(1, 1, H, W, device=device)
    mask[:, :, H//3:2*H//3, W//3:2*W//3] = 0.0        # fake collider hole

    # regimes in increasing-memory order: what step we PRETEND we're at
    regimes = [
        ('stage-A (transport)',      max(0, cfg.warp_gain_freeze_steps // 2)),  # gains frozen
        ('stage-A post-gain-unlock', cfg.warp_gain_freeze_steps + 10),
        ('stage-B (detail, L2)',     cfg.warp_stage_a_steps + 10),
        ('stage-B + GAN',            trainer.gan_start_step + trainer.gan_ramp_steps + 10),
    ]

    worst = 0.0
    failed = None
    recon = disc = float('nan')
    for name, step in regimes:
        trainer.global_step = step
        reset_peak(device)
        try:
            for _ in range(args.steps):
                frames = torch.randn(B, K1, C, H, W, device=device)
                recon, disc = trainer.step(frames, pixel_mask=mask)
            mem = peak_gb(device)
            worst = max(worst, mem if mem == mem else 0.0)
            ok = torch.isfinite(torch.tensor(recon)).item()
            adv = trainer._adv_weight()
            flag = 'OK ' if ok else 'NaN!'
            print(f'  [{flag}] {name:26s} peak={mem:6.2f}GB  recon={recon:.4f} '
                  f'disc={disc:.3f} adv_w={adv:.3f}')
            if not ok:
                failed = f'{name}: non-finite loss'
        except (RuntimeError, torch.OutOfMemoryError) as e:
            msg = str(e).split('\n')[0]
            print(f'  [FAIL] {name:26s} {type(e).__name__}: {msg}')
            failed = f'{name}: {type(e).__name__}'
            break

    tot = total_gb(device)
    print()
    if failed:
        print(f'PREFLIGHT FAILED at "{failed}".')
        if 'OutOfMemory' in failed:
            print(f'  peak reached the {tot:.0f}GB tile — lower --batch-size (linear), '
                  f'or drop warp_detail_res, before launching.')
        sys.exit(1)
    head = tot - worst if tot == tot else float('nan')
    print(f'PREFLIGHT PASSED.  worst-case peak={worst:.2f}GB'
          + (f' / {tot:.0f}GB  (headroom {head:.1f}GB)' if tot == tot else '')
          + f'\n  safe to launch at batch {args.batch_size}/rank.')
    if tot == tot and worst / tot > 0.9:
        print('  WARNING: >90% of tile used — thin margin; consider one notch lower.')


if __name__ == '__main__':
    main()

"""
Phase 1: train the latent autoencoder  L_t = encode(X_0, X_t),  X_t = decode(X_0, L_t).

Trained on (X_0, X_t) pairs (t sampled Uniform{1..ae_max_delta}) with a
reconstruction loss and, after `gan_start_step`, an adversarial loss.  No rollout,
no BPTT.

Run from the dfm/ root:
    python scripts/train_ae.py
    python scripts/train_ae.py --data /path/to/dataset --resume checkpoints_ae/ae_epoch009.pt
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import DFMConfig, AutoencoderTrainer
from dfm.data import FVMDataModule, build_renderer, load_pixel_mask
from dfm.profiling import LoopProfiler, make_profiler, finish_profiler
from dfm.distributed import init_distributed, is_main, allreduce_stats

_ROOT      = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
DEFAULT_TEST_DIR = _DATA_ROOT / 'test'
CKPT_DIR         = _ROOT / 'checkpoints_ae'
HYPERPARAMS      = _ROOT / 'hyperparams.json'

GAN_START_STEP        = 10_000
GAN_RAMP_STEPS        = 2_000
DISC_UPDATE_THRESHOLD = 0.5


def load_config() -> tuple[DFMConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    from dataclasses import fields
    valid_keys = {f.name for f in fields(DFMConfig)}
    m = {k: v for k, v in hp['model'].items() if k in valid_keys}
    return DFMConfig(**m), hp['training']


def get_device() -> torch.device:
    from dfm.distributed import pick_device
    return pick_device()



def main():
    p = argparse.ArgumentParser(description='Train latent autoencoder (phase 1)')
    p.add_argument('--data',       type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--test-data',  type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument('--resume',     type=str, default=None, nargs='?', const='latest')
    p.add_argument('--epochs',     type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None,
                   help='Override dataloader workers (0 avoids fork; use under tight SLURM --mem)')
    p.add_argument('--log-every',  type=int, default=50)
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile even if enabled in hyperparams.json')
    p.add_argument('--no-gan', action='store_true',
                   help='Disable the GAN completely')
    p.add_argument('--profile', type=int, default=0, metavar='N',
                   help='Profile the first N steps with torch.profiler, print a kernel breakdown, then exit')
    args = p.parse_args()

    rank, world, local_rank, device = init_distributed()
    if is_main():
        print(f'Device: {device}')
    cfg, train_hp = load_config()
    n_epochs   = args.epochs or train_hp['n_epochs']
    batch_size = args.batch_size or train_hp['batch_size']
    num_workers  = args.num_workers if args.num_workers is not None else train_hp.get('num_workers', 4)
    cache_frames = train_hp.get('cache_frames', False)

    # pred window = [X_0, X_1, ..., X_{ae_max_delta}]  → pairs sampled from it
    dm = FVMDataModule(args.data, n_context=cfg.n_context_frames, horizon=cfg.ae_max_delta,
                       batch_size=batch_size, num_workers=num_workers,
                       cache_frames=cache_frames, random_context=True)
    dm.setup()
    assert dm._dataset is not None
    steps_per_epoch = math.ceil(len(dm._dataset) / batch_size)
    total_steps     = steps_per_epoch * n_epochs
    mesh_dirs = []
    if (args.data / 'shared_mesh.pkl').exists():
        mesh_dirs.append(args.data)
    else:
        for p in args.data.iterdir():
            if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                mesh_dirs.append(p)
    if not mesh_dirs:
        raise RuntimeError(f'No shared_mesh.pkl found in {args.data} or its subdirectories')

    renderer   = build_renderer(mesh_dirs[0], cfg.img_hw)
    pixel_mask = load_pixel_mask(mesh_dirs[0], renderer, cfg.img_hw, frame_mask=cfg.frame_mask).to(device)

    val_dl = val_pm = None
    if args.test_data.exists():
        # num_workers=0 for validation — avoids forking dataloader workers against the
        # (large, CUDA-initialised) parent, which fails with ENOMEM under a memory cap.
        vdm = FVMDataModule(args.test_data, n_context=cfg.n_context_frames, horizon=cfg.ae_max_delta,
                            batch_size=batch_size, num_workers=0,
                            cache_frames=cache_frames, random_context=False,
                            mean=dm.mean, std=dm.std)
        vdm.setup()
        v_mesh_dirs = []
        if (args.test_data / 'shared_mesh.pkl').exists():
            v_mesh_dirs.append(args.test_data)
        else:
            for p in args.test_data.iterdir():
                if p.is_dir() and (p / 'shared_mesh.pkl').exists():
                    v_mesh_dirs.append(p)
        
        vr = build_renderer(v_mesh_dirs[0] if v_mesh_dirs else args.test_data, cfg.img_hw)
        val_pm = load_pixel_mask(v_mesh_dirs[0] if v_mesh_dirs else args.test_data, vr, cfg.img_hw, frame_mask=cfg.frame_mask).to(device)
        val_dl = vdm.val_dataloader()

    disable_gan = args.no_gan or train_hp.get('disable_gan', False)
    actual_gan_start = 10**9 if disable_gan else GAN_START_STEP
    trainer = AutoencoderTrainer(
        cfg, lr=train_hp['lr'], weight_decay=train_hp['weight_decay'],
        l1_weight=train_hp['l1_weight'], clip_grad=train_hp['clip_grad'],
        gan_start_step=actual_gan_start, gan_ramp_steps=GAN_RAMP_STEPS,
        disc_update_threshold=DISC_UPDATE_THRESHOLD, total_steps=total_steps,
        pixel_mask=pixel_mask,
        norm_stats=(dm.mean, dm.std),   # velocity denorm for the flow aux loss
    ).to(device)

    CKPT_DIR.mkdir(exist_ok=True)
    if args.resume == 'latest':
        cands = sorted(CKPT_DIR.glob('ae_*.pt'))
        args.resume = str(cands[-1]) if cands else None
    if args.resume:
        if is_main(): print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    trainer.wrap_ddp(device)

    # --- CUDA throughput: TF32 + torch.compile ---
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        if train_hp.get('compile', False) and not args.no_compile:
            print('Compiling autoencoder (first steps will be slow)...')
            # NOTE: leave the discriminator uncompiled — its spectral_norm layers
            # mutate the power-iteration buffers in place, and calling it twice
            # (real, then fake) before d_loss.backward() trips torch.compile's
            # saved-tensor version check ("modified by an inplace operation").
            trainer.ae = torch.compile(trainer.ae)

    n = lambda mod: sum(pp.numel() for pp in mod.parameters()) / 1e6
    print(f'Autoencoder:   {n(trainer.ae):.1f}M params')
    print(f'Discriminator: {n(trainer.discriminator):.1f}M params')
    start_epoch     = trainer.global_step // steps_per_epoch
    print(f'Dataset:       {len(dm._dataset)} sequences  (ae_max_delta={cfg.ae_max_delta})')
    if disable_gan:
        print('Curriculum:    GAN is DISABLED\n')
    else:
        print(f'Curriculum:    GAN activates at step {GAN_START_STEP}\n')

    log = None
    if is_main():
        log = open(CKPT_DIR / 'ae_loss_log.csv', 'a', newline='')
        w = csv.writer(log)
        if (CKPT_DIR / 'ae_loss_log.csv').stat().st_size == 0:
            w.writerow(['epoch', 'train_recon', 'val_recon']); log.flush()

    prof  = LoopProfiler(device)
    tprof = make_profiler(args.profile > 0, device)
    # Build the loader ONCE and reuse it across epochs: with persistent_workers the
    # workers are forked a single time at startup and stay alive, instead of
    # re-forking against an ever-larger parent each epoch (fork ENOMEM).
    train_dl = dm.train_dataloader()
    for epoch in range(start_epoch, n_epochs):
        rsum, rcnt = 0.0, 0
        for _, pred_b in train_dl:
            prof.data_ready()
            # incremental composition: consecutive-pair increments composed to the
            # map from X_0 over the WHOLE window (the one and only training mode)
            recon, disc = trainer.step(
                pred_b.to(device, non_blocking=True), pixel_mask=pixel_mask)
            prof.step_done(pred_b.shape[0])
            step = trainer.global_step
            if tprof is not None and step >= args.profile:
                finish_profiler(tprof, device, CKPT_DIR); return
            if not math.isfinite(recon):
                print(f'  [WARN] step {step}: NaN recon'); continue
            rsum += recon; rcnt += 1
            if step % args.log_every == 0:
                info = trainer.training_info()
                # do-nothing baseline for THIS batch: recon/base is the readable
                # convergence signal (raw recon is dominated by batch difficulty)
                base = trainer.persistence_baseline(
                    pred_b.to(device, non_blocking=True), pixel_mask=pixel_mask)
                ratio = recon / base if base > 1e-9 else float('nan')
                mh = getattr(trainer.ae, '_orig_mod', trainer.ae).decoder.map_head
                tm = getattr(mh, 'last_trans_mag', 0.0)
                cm = getattr(mh, 'last_curl_mag', 0.0)
                print(f'epoch {epoch:3d}  step {step:6d} | recon={recon:.4f} '
                      f'base={base:.4f} r/b={ratio:.2f} '
                      f'|trans|={tm:.4f} |curl|={cm:.4f}  '
                      f'disc={disc:.4f}  adv_w={info["adv_weight"]:.3f}  |  {prof.line()}')

            ckpt_after = train_hp.get('checkpoint_after', 0)
            if ckpt_after > 0 and step > 0 and step % ckpt_after == 0:
                if is_main():
                    path = CKPT_DIR / f'ae_step{step:06d}.pt'
                    trainer.save(str(path))
                    print(f'  [ckpt] {path.name}')

        train_r = rsum / rcnt if rcnt else float('nan')
        val_r = trainer.validate(val_dl, pixel_mask=val_pm) if val_dl else float('nan')
        train_r, val_r = allreduce_stats(train_r, val_r)
        train_r /= world
        val_r /= world

        if is_main():
            print(f'  [epoch {epoch:3d}] train_recon={train_r:.5f}  val_recon={val_r:.5f}')
            w.writerow([epoch, f'{train_r:.6f}', f'{val_r:.6f}']); log.flush()
            if (epoch + 1) % 2 == 0:
                path = CKPT_DIR / f'ae_epoch{epoch:03d}.pt'
                trainer.save(str(path)); print(f'  [ckpt] {path.name}')
    
    if is_main() and log is not None:
        log.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()

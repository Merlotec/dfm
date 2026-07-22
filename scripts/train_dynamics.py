"""
Phase 2: train the latent dynamics model on a frozen autoencoder's latents.

Loads the phase-1 AE (--ae), encodes ground-truth latents L_t = encode(X_0, X_t),
and trains the dynamics operator to predict L_{t+1} from L_t (teacher forced) —
no rollout, no BPTT.  Only the dynamics operator trains.

Run from the dfm/ root:
    python scripts/train_dynamics.py --ae checkpoints_ae/ae_epoch049.pt
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import DFMConfig, RolloutTrainer
from dfm.data import FVMDataModule, build_renderer, load_pixel_mask
from dfm.profiling import LoopProfiler, make_profiler, finish_profiler
from dfm.distributed import init_distributed, is_main, allreduce_stats

_ROOT      = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
DEFAULT_TEST_DIR = _DATA_ROOT / 'test'
CKPT_DIR         = _ROOT / 'checkpoints_dyn'
HYPERPARAMS      = _ROOT / 'hyperparams.json'


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
    p = argparse.ArgumentParser(description='Train latent dynamics (phase 2)')
    p.add_argument('--ae',         type=str, required=True, help='Phase-1 AE checkpoint')
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

    # pred window = [X_0, X_1, ..., X_{horizon_max}]  (the anchored sequence).
    dm = FVMDataModule(args.data, n_context=cfg.n_context_frames, horizon=cfg.horizon_max,
                       batch_size=batch_size, num_workers=num_workers,
                       cache_frames=cache_frames, random_context=True, return_index=False)
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
        vdm = FVMDataModule(args.test_data, n_context=cfg.n_context_frames, horizon=cfg.horizon_max,
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

    trainer = RolloutTrainer(
        cfg, lr=train_hp['lr'], weight_decay=train_hp['weight_decay'],
        clip_grad=train_hp['clip_grad'], total_steps=total_steps,
        pixel_mask=pixel_mask,
        latent_loss_weight=train_hp.get('latent_loss_weight'),
    ).to(device)

    print(f'latent_loss_weight: {trainer.latent_loss_weight}  '
          f'(model cfg {cfg.latent_loss_weight}, training section '
          f'{train_hp.get("latent_loss_weight")})')
    print(f'Loading frozen AE: {args.ae}')
    trainer.load_ae(args.ae)

    CKPT_DIR.mkdir(exist_ok=True)
    if args.resume == 'latest':
        cands = sorted(CKPT_DIR.glob('dyn_*.pt'))
        args.resume = str(cands[-1]) if cands else None
    if args.resume:
        if is_main(): print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    trainer.wrap_ddp(device)

    # --- CUDA throughput: TF32 + torch.compile (kernel fusion) ---
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        if train_hp.get('compile', False) and not args.no_compile:
            print('Compiling models (first steps will be slow)...')
            trainer.evo = torch.compile(trainer.evo)

    n = lambda mod: sum(pp.numel() for pp in mod.parameters()) / 1e6
    print(f'EvolutionOperator: {n(trainer.evo):.1f}M params')
    print(f'AE (frozen):       {n(trainer.ae):.1f}M params')
    start_epoch     = trainer.global_step // steps_per_epoch
    print(f'Dataset:        {len(dm._dataset)} sequences\n')

    log = None
    if is_main():
        log = open(CKPT_DIR / 'dyn_loss_log.csv', 'a', newline='')
        w = csv.writer(log)
        if (CKPT_DIR / 'dyn_loss_log.csv').stat().st_size == 0:
            w.writerow(['epoch', 'train_field_loss', 'train_latent_loss', 'val_field_loss',
                        'val_teacher_forced', 'val_persistence']); log.flush()

    prof  = LoopProfiler(device)
    tprof = make_profiler(args.profile > 0, device)
    train_dl = dm.train_dataloader()
    for epoch in range(start_epoch, n_epochs):
        # Reshuffle the DistributedSampler each epoch — without this every rank
        # replays the SAME shard order every epoch (no-op for the plain
        # RandomSampler of a single-process run, which has no set_epoch).
        if hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(epoch)
        fsum, lsum, count = 0.0, 0.0, 0
        for _, pred_b in train_dl:
            prof.data_ready()
            pred_b    = pred_b.to(device, non_blocking=True)
            field, latent = trainer.step(pred_b, pixel_mask=pixel_mask)
            prof.step_done(pred_b.shape[0])
            step = trainer.global_step
            if tprof is not None and step >= args.profile:
                finish_profiler(tprof, device, CKPT_DIR); return
            if not math.isfinite(field):
                print(f'  [WARN] step {step}: NaN field loss'); continue
            fsum += field; lsum += latent; count += 1
            if step % args.log_every == 0:
                # f/tf ~ 1 => at the frozen AE's floor (dynamics done);
                # f/b  is the phase-1-style persistence ratio.
                tf, base = trainer.reference_losses(pred_b, pixel_mask=pixel_mask,
                                                    K=trainer.last_K)
                f_tf = field / tf   if tf   > 1e-9 else float('nan')
                f_b  = field / base if base > 1e-9 else float('nan')
                print(f'epoch {epoch:3d}  step {step:6d} | field={field:.5f} '
                      f'tf={tf:.5f} base={base:.5f} f/tf={f_tf:.2f} f/b={f_b:.2f} '
                      f'latent={latent:.5f}  |  {prof.line()}')

            ckpt_after = train_hp.get('checkpoint_after', 0)
            if ckpt_after > 0 and step > 0 and step % ckpt_after == 0:
                if is_main():
                    path = CKPT_DIR / f'dyn_step{step:06d}.pt'
                    trainer.save(str(path))
                    print(f'  [ckpt] {path.name}')

        train_f = fsum / count if count else float('nan')
        train_l = lsum / count if count else float('nan')
        val_f, val_tf, val_base = (trainer.validate(val_dl, pixel_mask=val_pm)
                                   if val_dl else (float('nan'),) * 3)

        train_f, train_l, val_f, val_tf, val_base = allreduce_stats(
            train_f, train_l, val_f, val_tf, val_base)
        train_f /= world
        train_l /= world
        val_f /= world
        val_tf /= world
        val_base /= world

        if is_main():
            vf_tf = val_f / val_tf   if val_tf   > 1e-9 else float('nan')
            vf_b  = val_f / val_base if val_base > 1e-9 else float('nan')
            print(f'  [epoch {epoch:3d}] train_field={train_f:.5f} '
                  f'train_latent={train_l:.5f}  val_field={val_f:.5f} '
                  f'val_tf={val_tf:.5f} val_base={val_base:.5f} '
                  f'val_f/tf={vf_tf:.2f} val_f/b={vf_b:.2f}')
            w.writerow([epoch, f'{train_f:.6f}', f'{train_l:.6f}', f'{val_f:.6f}',
                        f'{val_tf:.6f}', f'{val_base:.6f}']); log.flush()
            if (epoch + 1) % 2 == 0:
                path = CKPT_DIR / f'dyn_epoch{epoch:03d}.pt'
                trainer.save(str(path)); print(f'  [ckpt] {path.name}')
    
    if is_main() and log is not None:
        log.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()

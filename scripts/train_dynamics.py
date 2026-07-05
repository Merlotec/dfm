"""
Phase 2: train the latent dynamics model on a frozen autoencoder's latents.

Loads the phase-1 AE (--ae), encodes ground-truth latents L_t = encode(X_0, X_t),
and trains the dynamics operator to predict L_{t+1} from L_t (teacher forced) —
no rollout, no BPTT.  Only the dynamics operator and the context encoder train.

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

from dfm import DFMConfig, LatentDynamicsTrainer
from dfm.data import FVMDataModule, build_renderer, load_pixel_mask
from dfm.profiling import LoopProfiler, make_profiler, finish_profiler

_ROOT      = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
DEFAULT_TEST_DIR = _DATA_ROOT / 'test'
CKPT_DIR         = _ROOT / 'checkpoints_dyn'
HYPERPARAMS      = _ROOT / 'hyperparams.json'


def load_config() -> tuple[DFMConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    m = {k: v for k, v in hp['model'].items() if not k.startswith('_')}
    return DFMConfig(**m), hp['training']


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    p = argparse.ArgumentParser(description='Train latent dynamics (phase 2)')
    p.add_argument('--ae',         type=str, required=True, help='Phase-1 AE checkpoint')
    p.add_argument('--data',       type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--test-data',  type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument('--resume',     type=str, default=None, nargs='?', const='latest')
    p.add_argument('--epochs',     type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--log-every',  type=int, default=50)
    p.add_argument('--evolve-state', action='store_true',
                   help='Also evolve the state embedding s_t in latent (optional second stream)')
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile even if enabled in hyperparams.json')
    p.add_argument('--profile', type=int, default=0, metavar='N',
                   help='Profile the first N steps with torch.profiler, print a kernel breakdown, then exit')
    args = p.parse_args()

    device = get_device()
    print(f'Device: {device}')
    cfg, train_hp = load_config()
    if args.evolve_state:
        cfg.evolve_state = True
    print(f'evolve_state: {cfg.evolve_state}')
    n_epochs   = args.epochs or train_hp['n_epochs']
    batch_size = args.batch_size or train_hp['batch_size']
    num_workers  = train_hp.get('num_workers', 4)
    cache_frames = train_hp.get('cache_frames', False)

    # pred window = [X_0, X_1, ..., X_{horizon_max}]  (the anchored sequence).
    # return_index enables the frozen-AE latent cache in the trainer.
    dm = FVMDataModule(args.data, n_context=cfg.n_context_frames, horizon=cfg.horizon_max,
                       batch_size=batch_size, num_workers=num_workers,
                       cache_frames=cache_frames, random_context=True, return_index=True)
    dm.setup()
    assert dm._dataset is not None
    steps_per_epoch = math.ceil(len(dm._dataset) / batch_size)
    total_steps     = steps_per_epoch * n_epochs
    renderer   = build_renderer(args.data, (cfg.img_size, cfg.img_size))
    pixel_mask = load_pixel_mask(args.data, renderer, (cfg.img_size, cfg.img_size)).to(device)

    val_dl = val_pm = None
    if args.test_data.exists():
        vdm = FVMDataModule(args.test_data, n_context=cfg.n_context_frames, horizon=cfg.horizon_max,
                            batch_size=batch_size, num_workers=num_workers,
                            cache_frames=cache_frames, random_context=False,
                            mean=dm.mean, std=dm.std)
        vdm.setup()
        vr = build_renderer(args.test_data, (cfg.img_size, cfg.img_size))
        val_pm = load_pixel_mask(args.test_data, vr, (cfg.img_size, cfg.img_size)).to(device)
        val_dl = vdm.val_dataloader()

    trainer = LatentDynamicsTrainer(
        cfg, lr=train_hp['lr'], weight_decay=train_hp['weight_decay'],
        clip_grad=train_hp['clip_grad'], total_steps=total_steps,
        pixel_mask=pixel_mask,
    ).to(device)

    print(f'Loading frozen AE: {args.ae}')
    trainer.load_ae(args.ae)

    CKPT_DIR.mkdir(exist_ok=True)
    if args.resume == 'latest':
        cands = sorted(CKPT_DIR.glob('dyn_*.pt'))
        args.resume = str(cands[-1]) if cands else None
    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    # --- CUDA throughput: TF32 + torch.compile (kernel fusion) ---
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        if train_hp.get('compile', False) and not args.no_compile:
            print('Compiling models (first steps will be slow)...')
            trainer.dynamics        = torch.compile(trainer.dynamics)
            trainer.context_encoder = torch.compile(trainer.context_encoder)
            trainer.ae              = torch.compile(trainer.ae)

    n = lambda mod: sum(pp.numel() for pp in mod.parameters()) / 1e6
    print(f'Dynamics:       {n(trainer.dynamics):.1f}M params')
    print(f'ContextEncoder: {n(trainer.context_encoder):.1f}M params')
    print(f'AE (frozen):    {n(trainer.ae):.1f}M params')
    start_epoch     = trainer.global_step // steps_per_epoch
    print(f'Dataset:        {len(dm._dataset)} sequences  (latent MSE, teacher forced, no BPTT)\n')

    log = open(CKPT_DIR / 'dyn_loss_log.csv', 'a', newline='')
    w = csv.writer(log)
    if (CKPT_DIR / 'dyn_loss_log.csv').stat().st_size == 0:
        w.writerow(['epoch', 'train_latent_mse', 'val_latent_mse']); log.flush()

    prof  = LoopProfiler(device)
    tprof = make_profiler(args.profile > 0, device)
    for epoch in range(start_epoch, n_epochs):
        lsum, lcnt = 0.0, 0
        for idx_b, context_b, pred_b in dm.train_dataloader():
            prof.data_ready()
            context_frames = [context_b[:, t].to(device) for t in range(context_b.shape[1])]
            pred_frames    = [pred_b[:, t].to(device)    for t in range(pred_b.shape[1])]
            loss = trainer.step(context_frames, pred_frames, pixel_mask=pixel_mask, index=idx_b)
            prof.step_done(pred_b.shape[0])
            step = trainer.global_step
            if tprof is not None and step >= args.profile:
                finish_profiler(tprof, device, CKPT_DIR); return
            if not math.isfinite(loss):
                print(f'  [WARN] step {step}: NaN latent loss'); continue
            lsum += loss; lcnt += 1
            if step % args.log_every == 0:
                print(f'epoch {epoch:3d}  step {step:6d} | latent_mse={loss:.5f}  |  {prof.line()}')

        train_l = lsum / lcnt if lcnt else float('nan')
        val_l = trainer.validate(val_dl, pixel_mask=val_pm) if val_dl else float('nan')
        print(f'  [epoch {epoch:3d}] train_latent_mse={train_l:.5f}  val_latent_mse={val_l:.5f}')
        w.writerow([epoch, f'{train_l:.6f}', f'{val_l:.6f}']); log.flush()
        if (epoch + 1) % 2 == 0:
            path = CKPT_DIR / f'dyn_epoch{epoch:03d}.pt'
            trainer.save(str(path)); print(f'  [ckpt] {path.name}')
    log.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()

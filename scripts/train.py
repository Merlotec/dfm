"""
Full-dataset training for HFM-1D (RolloutGANTrainer).

Two-stage curriculum, managed automatically:
  Stage 1 (steps 0 → gan_start_step):   reconstruction only
  Stage 2 (steps gan_start_step → end): GAN active
    adv_weight ramps 0 → cfg.disc_adv_weight over gan_ramp_steps steps.

Run from the hfm1d/ root:
    python scripts/train.py
    python scripts/train.py --data /path/to/dataset
    python scripts/train.py --resume checkpoints/train_epoch009.pt
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfm1d import HFM1DConfig, RolloutGANTrainer
from hfm1d.data import FVMDataModule, build_renderer, load_pixel_mask

_ROOT      = Path(__file__).resolve().parents[1]
_DATA_ROOT = _ROOT.parent / 'data'

DEFAULT_DATA_DIR = _DATA_ROOT / 'fvm_gen_datasets'
DEFAULT_TEST_DIR = _DATA_ROOT / 'test'
CKPT_DIR         = _ROOT / 'checkpoints'
HYPERPARAMS      = _ROOT / 'hyperparams.json'

GAN_START_STEP        = 10_000
GAN_RAMP_STEPS        = 2_000
DISC_UPDATE_THRESHOLD = 0.5


def load_config() -> tuple[HFM1DConfig, dict]:
    with open(HYPERPARAMS) as f:
        hp = json.load(f)
    m, t = hp['model'], hp['training']
    cfg = HFM1DConfig(
        img_size         = m['img_size'],
        in_channels      = m['in_channels'],
        patch_px         = m['patch_px'],
        d_model          = m['d_model'],
        n_slots          = m['n_slots'],
        n_heads          = m['n_heads'],
        n_enc_layers     = m['n_enc_layers'],
        n_evo_layers     = m['n_evo_layers'],
        integrator       = m['integrator'],
        max_rollout      = m['max_rollout'],
        reencode_every   = m['reencode_every'],
        skip_ch          = m['skip_ch'],
        horizon          = m['horizon'],
        horizon_gamma    = m['horizon_gamma'],
        noise_std        = m['noise_std'],
        mlp_ratio        = m['mlp_ratio'],
        dropout          = m['dropout'],
        n_context_frames = m['n_context_frames'],
        ctx_patch_px     = m['ctx_patch_px'],
        d_ctx            = m['d_ctx'],
        n_ctx_tokens     = m['n_ctx_tokens'],
        n_ctx_layers     = m['n_ctx_layers'],
        n_ctx_heads      = m['n_ctx_heads'],
        disc_dim         = m['disc_dim'],
        disc_adv_weight  = m['disc_adv_weight'],
        disc_lr          = m['disc_lr'],
    )
    return cfg, t


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Train HFM-1D on fluid simulation data')
    parser.add_argument('--data',      type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--test-data', type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument('--resume',    type=str, default=None, nargs='?', const='latest')
    parser.add_argument('--epochs',    type=int, default=None)
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}')

    cfg, train_hp = load_config()
    n_epochs = args.epochs or train_hp['n_epochs']
    # history frames + seed frame + horizon target frames
    seq_len  = cfg.n_context_frames + 1 + cfg.horizon

    dm = FVMDataModule(
        data_dir    = args.data,
        seq_len     = seq_len,
        batch_size  = train_hp['batch_size'],
        num_workers = 4,
    )
    dm.setup()

    renderer   = build_renderer(args.data, (cfg.img_size, cfg.img_size))
    pixel_mask = load_pixel_mask(args.data, renderer, (cfg.img_size, cfg.img_size)).to(device)

    val_dl = val_pixel_mask = None
    if args.test_data.exists():
        val_dm = FVMDataModule(
            data_dir    = args.test_data,
            seq_len     = seq_len,
            batch_size  = train_hp['batch_size'],
            num_workers = 4,
            mean        = dm.mean,
            std         = dm.std,
        )
        val_dm.setup()
        val_renderer   = build_renderer(args.test_data, (cfg.img_size, cfg.img_size))
        val_pixel_mask = load_pixel_mask(
            args.test_data, val_renderer, (cfg.img_size, cfg.img_size)
        ).to(device)
        val_dl = val_dm.val_dataloader()

    trainer = RolloutGANTrainer(
        cfg,
        lr                    = train_hp['lr'],
        weight_decay          = train_hp['weight_decay'],
        l1_weight             = train_hp['l1_weight'],
        clip_grad             = train_hp['clip_grad'],
        gan_start_step        = GAN_START_STEP,
        gan_ramp_steps        = GAN_RAMP_STEPS,
        disc_update_threshold = DISC_UPDATE_THRESHOLD,
        pixel_mask            = pixel_mask,
    )
    trainer.to(device)

    if args.resume == 'latest':
        candidates = sorted(CKPT_DIR.glob('train_*.pt'))
        if not candidates:
            print(f'No checkpoints found in {CKPT_DIR}')
            sys.exit(1)
        args.resume = str(candidates[-1])
    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load(str(args.resume))

    n_gen  = sum(p.numel() for p in trainer.model.parameters())           / 1e6
    n_ctx  = sum(p.numel() for p in trainer.context_encoder.parameters()) / 1e6
    n_disc = sum(p.numel() for p in trainer.discriminator.parameters())   / 1e6
    print(f'Generator:       {n_gen:.1f}M params')
    print(f'ContextEncoder:  {n_ctx:.1f}M params')
    print(f'Discriminator:   {n_disc:.1f}M params')
    assert dm._dataset is not None
    steps_per_epoch = math.ceil(len(dm._dataset) / train_hp['batch_size'])
    start_epoch     = trainer.global_step // steps_per_epoch
    print(f'Dataset:         {len(dm._dataset)} sequences  (seq_len={seq_len}, horizon={cfg.horizon})')
    print(f'Curriculum:      GAN activates at step {GAN_START_STEP}\n')

    CKPT_DIR.mkdir(exist_ok=True)
    loss_log_path = CKPT_DIR / 'loss_log.csv'
    write_header  = not loss_log_path.exists()
    loss_csv      = open(loss_log_path, 'a', newline='')
    loss_writer   = csv.writer(loss_csv)
    if write_header:
        loss_writer.writerow(['epoch', 'train_loss', 'val_loss'])
        loss_csv.flush()

    nan_streak = 0
    for epoch in range(start_epoch, n_epochs):
        recon_sum, recon_cnt = 0.0, 0
        for batch in dm.train_dataloader():
            frames = [batch[:, t].to(device) for t in range(batch.shape[1])]
            recon, disc = trainer.step(frames, pixel_mask=pixel_mask)
            info = trainer.training_info()
            step = info['global_step']

            if not math.isfinite(recon):
                nan_streak += 1
                print(f'  [WARN] step {step}: NaN/inf loss (streak={nan_streak})')
                if nan_streak >= 10:
                    trainer.save(str(CKPT_DIR / f'train_EMERGENCY_step{step:06d}.pt'))
                    sys.exit(1)
                continue
            nan_streak = 0

            recon_sum += recon
            recon_cnt += 1
            if step % args.log_every == 0:
                print(f'epoch {epoch:3d}  step {step:6d} | recon={recon:.4f}  '
                      f'disc={disc:.4f}  adv_w={info["adv_weight"]:.3f}')

        train_loss = recon_sum / recon_cnt if recon_cnt else float('nan')
        val_loss   = trainer.validate(val_dl, pixel_mask=val_pixel_mask) if val_dl else float('nan')
        print(f'  [epoch {epoch:3d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')
        loss_writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}'])
        loss_csv.flush()

        if (epoch + 1) % 2 == 0:
            path = CKPT_DIR / f'train_epoch{epoch:03d}.pt'
            trainer.save(str(path))
            print(f'  [ckpt] {path.name}')

    loss_csv.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()

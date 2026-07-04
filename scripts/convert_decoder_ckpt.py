"""
Convert a checkpoint trained with the OLD single-block decoder
(`decoder.slot_read.*`, one cross-attention) to the CURRENT decoder, which
stacks `n_dec_layers` blocks of [cross-attention → self-attention].

Mapping (for n_dec_layers=1):
  decoder.slot_read.*   →   decoder.layers.0.cross.*        (weights carried over)
  decoder.layers.0.self_attn.*   →   identity-initialised   (zero output projections)

Identity-initialising the new self-attention means the converted model produces
the *same* output as the original decoder at load time; the self-attention is
then learned from there.  The stored config is updated to n_dec_layers=1 and the
(now parameter-mismatched) generator optimizer state is dropped.

Usage:
    python scripts/convert_decoder_ckpt.py <src.pt> <dst.pt> [--n-dec-layers 1]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfm import HFM1D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--n-dec-layers', type=int, default=1)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    cfg.n_dec_layers = args.n_dec_layers
    print(f'Target: n_dec_layers={cfg.n_dec_layers}  d_model={cfg.d_model}  '
          f'n_slots={cfg.n_slots}  global_step={ckpt.get("global_step")}')

    # --- remap the old single cross-attn block onto layer 0's cross-attn ---
    old = ckpt['model']
    OLD_P, NEW_P = 'decoder.slot_read.', 'decoder.layers.0.cross.'
    remapped = {}
    n_remap = 0
    for k, v in old.items():
        if k.startswith(OLD_P):
            remapped[NEW_P + k[len(OLD_P):]] = v
            n_remap += 1
        else:
            remapped[k] = v
    print(f'Remapped {n_remap} decoder.slot_read.* → decoder.layers.0.cross.*')

    # --- build target model; identity-init every new self-attention block ---
    model = HFM1D(cfg)
    for blk in model.decoder.layers:
        sa = blk.self_attn
        nn.init.zeros_(sa.attn.out_proj.weight)   # attention residual → 0
        nn.init.zeros_(sa.attn.out_proj.bias)
        last = sa.ffn.net[3]                        # FFN final linear → 0
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    non_sa_missing = [m for m in missing if '.self_attn.' not in m]
    print(f'Loaded. self_attn keys kept at identity init: '
          f'{sum(1 for m in missing if ".self_attn." in m)}')
    if non_sa_missing:
        print(f'  [warn] unexpected MISSING (non self_attn): {non_sa_missing}')
    if unexpected:
        print(f'  [warn] UNEXPECTED keys in checkpoint: {unexpected}')

    # --- sanity: each self_attn block must be a strict identity ---
    with torch.no_grad():
        q = torch.randn(2, cfg.n_patch ** 2, cfg.d_model)
        for i, blk in enumerate(model.decoder.layers):
            out = blk.self_attn(q)
            assert torch.allclose(out, q, atol=1e-6), f'self_attn[{i}] not identity'
    print('Verified: all self_attn blocks are identity at load.')

    # --- write converted checkpoint (drop param-mismatched gen_optimizer) ---
    new_ckpt = {
        'model':           model.state_dict(),
        'context_encoder': ckpt['context_encoder'],
        'discriminator':   ckpt['discriminator'],
        'scheduler':       ckpt.get('scheduler'),
        'cfg':             cfg,
        'global_step':     ckpt.get('global_step', 0),
    }
    if 'disc_optimizer' in ckpt:      # discriminator unchanged → state still valid
        new_ckpt['disc_optimizer'] = ckpt['disc_optimizer']
    # gen_optimizer intentionally omitted (model gained self_attn params).

    torch.save(new_ckpt, args.dst)
    print(f'Wrote {args.dst}')


if __name__ == '__main__':
    main()

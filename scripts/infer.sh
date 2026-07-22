#!/bin/bash
# DFM inference on the held-out test set (../data/test).
#
#   scripts/infer.sh <checkpoint.pt> [extra args]
#
# Phase-1 AE checkpoint  -> teacher-forced reconstruction:
#   scripts/infer.sh checkpoints_ae/ae_epoch009.pt --n-predict 10 --n-runs 4
# Phase-2 evo checkpoint -> autonomous rollout (needs the frozen AE too):
#   scripts/infer.sh checkpoints_dyn/dyn_latest.pt --ae checkpoints_ae/ae_latest.pt
#
# Outputs frames_*.npy + images/ + metrics.csv per run under out/infer/<run>/.
set -euo pipefail
CKPT="${1:?usage: infer.sh <checkpoint.pt> [--ae AE.pt] [--n-predict N] [--n-runs N] ...}"
shift
cd "$(dirname "$0")/.."
exec python scripts/infer.py --checkpoint "$CKPT" --data ../data/test \
     --out-dir out/infer "$@"

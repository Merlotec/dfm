#!/bin/bash
# 30-second preflight on ONE PVC tile: forces every curriculum boundary (stage-B
# DetailHead, GAN) on at your production batch and reports peak XPU memory, so a
# batch that would OOM 4 epochs in fails HERE instead — in a 10-minute slot.
#
#   sbatch scripts/dawn_preflight.slurm 6      # 6 = per-rank batch to test
#
# Single-process on purpose: it measures PER-RANK activation memory (the thing
# that OOMs), so no CCL / gloo / DDP — none of the distributed machinery runs.
#
#SBATCH --job-name=dfm-preflight
#SBATCH --account=AIRR-P67-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --output=logs/dfm_preflight.out
#SBATCH --error=logs/dfm_preflight.err
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs

BATCH="${1:-6}"                                # per-rank batch to test (sbatch arg)

# ---- same environment as dawn_task.sh (pip oneAPI runtime, one tile) ---------
module purge
module load default-dawn                       # base env + level-zero driver
source "$HOME/rds/hpc-work/venvs/flsim-xpu/bin/activate"   # EDIT to your env
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib64:${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0                      # this job sees exactly one tile

cd "$SLURM_SUBMIT_DIR"
echo "preflight: batch=${BATCH}/rank"
srun --ntasks=1 python scripts/preflight.py --batch-size "${BATCH}"

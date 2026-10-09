#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
SEEDS=(3101 3102 3103 3104 3105)
INDEX="${SLURM_ARRAY_TASK_ID:?}"
test -f "${RUN_ROOT:?}/analysis/classical/FROZEN"
test -f "$RUN_ROOT/analysis/frozen_neural.FROZEN"
python collect_profiles.py --profile ood_arxiv \
  --physical-seed "${SEEDS[$INDEX]}" --seed-index "$INDEX" \
  --batches 6 \
  --shared-data-root "${DEEPSC_SHARED_ROOT:?}" \
  --vocab-file "$DEEPSC_SHARED_ROOT/shared_vocab.json" \
  --checkpoint-c "$RUN_ROOT/checkpoints/C/last.pth" \
  --checkpoint-d "$RUN_ROOT/checkpoints/D/last.pth" \
  --output-root "$RUN_ROOT/traces/holdout" --device cuda

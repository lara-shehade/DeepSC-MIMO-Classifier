#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
SEEDS=(3101 3102 3103 3104 3105)
TASK="${SLURM_ARRAY_TASK_ID:?}"
CONFIG_INDEX=$((TASK / 5))
SEED_INDEX=$((TASK % 5))
LINE=$((CONFIG_INDEX + 2))
IFS=, read -r ARCHITECTURE STAGE < <(sed -n "${LINE}p" config/neural_grid.csv)
OUTPUT="${RUN_ROOT:?}/analysis/neural_folds/${ARCHITECTURE}_${STAGE}_seed${SEEDS[$SEED_INDEX]}.json"
python analysis/neural_diagnostics.py --mode fold \
  --trace-root "$RUN_ROOT/traces/development" \
  --architecture "$ARCHITECTURE" --stage "$STAGE" \
  --held-out-seed "${SEEDS[$SEED_INDEX]}" --device cuda --output "$OUTPUT"

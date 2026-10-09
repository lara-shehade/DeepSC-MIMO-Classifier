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
PROFILE_INDEX=$((TASK / 5))
SEED_INDEX=$((TASK % 5))
LINE=$((PROFILE_INDEX + 2))
IFS=, read -r PROFILE SEVERITY < <(sed -n "${LINE}p" config/development_profiles.csv)
ARGS=(
  --profile "$PROFILE" --physical-seed "${SEEDS[$SEED_INDEX]}"
  --seed-index "$SEED_INDEX" --shared-data-root "${DEEPSC_SHARED_ROOT:?}"
  --vocab-file "$DEEPSC_SHARED_ROOT/shared_vocab.json"
  --checkpoint-c "${RUN_ROOT:?}/checkpoints/C/last.pth"
  --checkpoint-d "$RUN_ROOT/checkpoints/D/last.pth"
  --output-root "$RUN_ROOT/traces/development" --device cuda
)
if [[ -n "$SEVERITY" ]]; then ARGS+=(--severity "$SEVERITY"); fi
python collect_profiles.py "${ARGS[@]}"

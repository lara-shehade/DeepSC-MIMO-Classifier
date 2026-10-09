#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
test -s "${RUN_ROOT:?}/analysis/development_features.csv"
if [[ "${CLASSICAL_RECOVERY_MODE:?}" == fold ]]; then
  python -m server.classical_recovery "$RUN_ROOT/analysis/development_features.csv" \
    --output-root "$RUN_ROOT/analysis/classical" --mode fold \
    --fold-index "${SLURM_ARRAY_TASK_ID:?}"
elif [[ "$CLASSICAL_RECOVERY_MODE" == finalize ]]; then
  python -m server.classical_recovery "$RUN_ROOT/analysis/development_features.csv" \
    --output-root "$RUN_ROOT/analysis/classical" --mode finalize
else
  echo "Unknown CLASSICAL_RECOVERY_MODE: $CLASSICAL_RECOVERY_MODE" >&2
  exit 2
fi

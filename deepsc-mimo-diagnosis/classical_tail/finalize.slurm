#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00
set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${RECOVERY_ROOT:?}/code:${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m classical_tail.recover finalize \
  --features "${RUN_ROOT:?}/analysis/development_features.csv" \
  --output-root "$RUN_ROOT/analysis/classical" --recovery-root "$RECOVERY_ROOT"

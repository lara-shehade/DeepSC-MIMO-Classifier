#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${FREEZE_ROOT:?}/code:${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m classical_freeze_recovery.recover chunk \
  --features "${RUN_ROOT:?}/analysis/development_features.csv" \
  --output-root "$RUN_ROOT/analysis/classical" \
  --recovery-root "$FREEZE_ROOT" \
  --chunk-index "${FREEZE_CHUNK:?}"

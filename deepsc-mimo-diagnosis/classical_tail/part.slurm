#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00
set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
export PYTHONPATH="${RECOVERY_ROOT:?}/code:${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ "${TAIL_PILOT:-0}" == 1 ]]; then
  TASK=0
else
  TASK="${SLURM_ARRAY_TASK_ID:?Tail array task ID missing}"
fi
[[ "$TASK" =~ ^[0-8]$ ]] || { echo "Invalid tail task: $TASK" >&2; exit 2; }
python -m classical_tail.recover part \
  --features "${RUN_ROOT:?}/analysis/development_features.csv" \
  --output-root "$RUN_ROOT/analysis/classical" \
  --recovery-root "$RECOVERY_ROOT" --task-index "$TASK"

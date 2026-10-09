#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
: "${PROJECT_ROOT:?}" "${RUN_ROOT:?}" "${STUDY_ROOT:?}"
(cd "$STUDY_ROOT" && sha256sum -c code.sha256 >/dev/null)
sha256sum -c "$STUDY_ROOT/dependency.sha256" >/dev/null
cd "$PROJECT_ROOT"
export PYTHONPATH="$STUDY_ROOT/code:$PROJECT_ROOT"
INDEX="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:?}}"
python -m stage_subset.ablation --mode fold \
  --trace-root "$RUN_ROOT/traces/development" \
  --output-root "$STUDY_ROOT/results" \
  --task-index "$INDEX" --device cuda

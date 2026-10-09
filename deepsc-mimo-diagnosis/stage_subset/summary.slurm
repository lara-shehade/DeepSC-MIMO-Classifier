#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

set -euo pipefail
module purge
module load python/ai-4
: "${PROJECT_ROOT:?}" "${RUN_ROOT:?}" "${STUDY_ROOT:?}"
(cd "$STUDY_ROOT" && sha256sum -c code.sha256 >/dev/null)
sha256sum -c "$STUDY_ROOT/dependency.sha256" >/dev/null
cd "$PROJECT_ROOT"
export PYTHONPATH="$STUDY_ROOT/code:$PROJECT_ROOT"
python -m stage_subset.ablation --mode summary \
  --output-root "$STUDY_ROOT/results" \
  --singleton-root "$RUN_ROOT/analysis/neural_folds" --device cpu

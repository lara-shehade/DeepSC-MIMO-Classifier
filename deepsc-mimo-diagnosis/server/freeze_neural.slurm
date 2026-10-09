#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
python analysis/neural_diagnostics.py --mode freeze \
  --trace-root "$RUN_ROOT/traces/development" \
  --fold-root "$RUN_ROOT/analysis/neural_folds" \
  --device cuda --output "$RUN_ROOT/analysis/frozen_neural.pt"

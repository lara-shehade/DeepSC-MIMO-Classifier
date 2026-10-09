#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
mkdir -p "${RUN_ROOT:?}/analysis"
python analysis/extract_sentence_features.py "$RUN_ROOT/traces/development" \
  --output "$RUN_ROOT/analysis/development_features.csv"
python analysis/first_divergence.py "$RUN_ROOT/traces/development" \
  --output "$RUN_ROOT/analysis/first_divergence.json"
python analysis/classical_diagnostics.py "$RUN_ROOT/analysis/development_features.csv" \
  --output-root "$RUN_ROOT/analysis/classical"

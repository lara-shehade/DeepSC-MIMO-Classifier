#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
python analysis/extract_sentence_features.py "$RUN_ROOT/traces/holdout" \
  --include-holdout --output "$RUN_ROOT/analysis/holdout_features.csv"
python analysis/evaluate_holdout.py \
  --holdout-features "$RUN_ROOT/analysis/holdout_features.csv" \
  --holdout-traces "$RUN_ROOT/traces/holdout" \
  --classical-model "$RUN_ROOT/analysis/classical/frozen_classical.joblib" \
  --neural-model "$RUN_ROOT/analysis/frozen_neural.pt" \
  --device cuda --output "$RUN_ROOT/analysis/arxiv_holdout.json"

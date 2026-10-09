#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
export PYTHONPATH=${RUN_ROOT}/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ${RUN_ROOT}/code
python ${RUN_ROOT}/paper_extraction_v1/code/fusion_probs.py --mode analyze --probs-dir ${RUN_ROOT}/paper_extraction_v1/probs \
  --original-folds ${RUN_ROOT}/analysis/neural_folds \
  --batch-channel ${RUN_ROOT}/paper_extraction_v1/metrics/batch_channel.csv --output-dir ${RUN_ROOT}/paper_extraction_v1/analysis

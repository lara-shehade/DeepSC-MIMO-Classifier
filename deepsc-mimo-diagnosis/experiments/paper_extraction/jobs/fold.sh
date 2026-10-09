#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
export PYTHONPATH=${RUN_ROOT}/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ${RUN_ROOT}/code
SEEDS=(3101 3102 3103 3104 3105)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
python ${RUN_ROOT}/paper_extraction_v1/code/fusion_probs.py --mode fold --trace-root ${RUN_ROOT}/traces/development \
  --held-out-seed $S --device cpu --output ${RUN_ROOT}/paper_extraction_v1/probs/fusion_probs_seed$S.csv

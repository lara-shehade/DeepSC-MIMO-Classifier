#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
export PYTHONPATH=${RUN_ROOT}/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ${RUN_ROOT}/code
python ${RUN_ROOT}/paper_extraction_v1/code/paper_metrics.py --run-root ${RUN_ROOT} --shared-root ${DEEPSC_SHARED_ROOT} --output-dir ${RUN_ROOT}/paper_extraction_v1/metrics

#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
(cd ${RUN_ROOT}/correction_v1/code && sha256sum -c ../code.sha256 >/dev/null)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ${RUN_ROOT}/correction_v1/code
echo "RUN_ROOT=${RUN_ROOT}  OUT=${RUN_ROOT}/correction_v1  host=$(hostname)"
SEEDS=(3101 3102 3103 3104 3105)
python run_fold.py --run-root ${RUN_ROOT} --shared-root ${DEEPSC_SHARED_ROOT} --output ${RUN_ROOT}/correction_v1 --probs-dir ${RUN_ROOT}/paper_extraction_v1/probs --held-out-seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} --threads $SLURM_CPUS_PER_TASK

#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
(cd ${RUN_ROOT}/correction_v1/code && sha256sum -c ../code.sha256 >/dev/null)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ${RUN_ROOT}/correction_v1/code
echo "RUN_ROOT=${RUN_ROOT}  OUT=${RUN_ROOT}/correction_v1  host=$(hostname)"
python make_dd.py --run-root ${RUN_ROOT} --shared-root ${DEEPSC_SHARED_ROOT} --output ${RUN_ROOT}/correction_v1 --batches 10

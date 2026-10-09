#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
(cd ${RUN_ROOT}/correction_v1/code && sha256sum -c ../code.sha256 >/dev/null)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ${RUN_ROOT}/correction_v1/code
echo "RUN_ROOT=${RUN_ROOT}  OUT=${RUN_ROOT}/correction_v1  host=$(hostname)"
python summarize.py --output ${RUN_ROOT}/correction_v1
cd ${RUN_ROOT} && tar -czf ${SCRATCH_ROOT}/correction_v1_results.tgz --exclude=correction_v1/code correction_v1
echo "bundle: ${SCRATCH_ROOT}/correction_v1_results.tgz"

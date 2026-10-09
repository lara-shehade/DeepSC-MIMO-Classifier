#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
python server/final_summary.py "${RUN_ROOT:?}"

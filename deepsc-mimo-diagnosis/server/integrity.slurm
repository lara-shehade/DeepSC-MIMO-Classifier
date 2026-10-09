#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"
mkdir -p "${RUN_ROOT:?}/integrity"
python server/check_cd_integrity.py \
  --shared-data-root "${DEEPSC_SHARED_ROOT:?}" \
  --vocab-file "$DEEPSC_SHARED_ROOT/shared_vocab.json" \
  --checkpoint-c "$RUN_ROOT/checkpoints/C/last.pth" \
  --checkpoint-d "$RUN_ROOT/checkpoints/D/last.pth" \
  --device cuda \
  --output "$RUN_ROOT/integrity/cd_integrity.json"

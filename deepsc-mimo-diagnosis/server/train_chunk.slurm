#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:50:00

set -euo pipefail
module purge
module load python/ai-4
cd "${PROJECT_ROOT:?}"

MODEL_ID="${MODEL_ID:?}"
MODEL_SEED="${MODEL_SEED:?}"
CHUNK_INDEX="${CHUNK_INDEX:?}"
CHECKPOINT_DIR="${RUN_ROOT:?}/checkpoints/${MODEL_ID}"
TARGET_EPOCH=$((CHUNK_INDEX * 10))
ARGS=(
  --data-root "${DEEPSC_SHARED_ROOT:?}/europarl"
  --vocab-file "$DEEPSC_SHARED_ROOT/shared_vocab.json"
  --device cuda --epochs "$TARGET_EPOCH" --batch-size 128
  --seed "$MODEL_SEED" --channel-seed "$MODEL_SEED"
  --fading-scope batch --profile baseline --equalizer lmmse --covariance system
  --checkpoint-path "$CHECKPOINT_DIR"
)
if [[ "$CHUNK_INDEX" -gt 1 ]]; then
  test -f "$CHECKPOINT_DIR/last.pth"
  ARGS+=(--resume-checkpoint "$CHECKPOINT_DIR/last.pth")
fi
python main.py "${ARGS[@]}"
python - "$CHECKPOINT_DIR/last.pth" "$TARGET_EPOCH" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = int(sys.argv[2])
if int(checkpoint.get("epoch", -1)) != expected:
    raise SystemExit("checkpoint epoch mismatch")
if "rng_state" not in checkpoint:
    raise SystemExit("checkpoint lacks exact-resume RNG state")
print("verified epoch", expected)
PY

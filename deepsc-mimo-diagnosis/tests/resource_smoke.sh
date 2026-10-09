#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
TEMP="$(mktemp -d)"
trap 'rm -rf -- "$TEMP"' EXIT
mkdir -p "$TEMP/bin" "$TEMP/runs" "$TEMP/shared"
cp "$SOURCE/tests/mock_sbatch.sh" "$TEMP/bin/sbatch"
cp "$SOURCE/tests/mock_scheduler.sh" "$TEMP/bin/squeue"
cp "$SOURCE/tests/mock_scheduler.sh" "$TEMP/bin/scontrol"
printf '#!/bin/bash\nexit 0\n' > "$TEMP/bin/flock"
chmod +x "$TEMP/bin/sbatch" "$TEMP/bin/squeue" "$TEMP/bin/scontrol" "$TEMP/bin/flock"
export PATH="$TEMP/bin:$PATH"
export MOCK_COUNT="$TEMP/count" MOCK_LOG="$TEMP/sbatch.log" MOCK_MEMORY="$TEMP/memory.log"
export DEEPSC_SHARED_ROOT="$TEMP/shared" DEEPSC_RUNS_ROOT="$TEMP/runs"
touch "$MOCK_MEMORY"
bash "$SOURCE/server/launch_pipeline.sh" preflight resources > "$TEMP/preflight.out"
export MOCK_RUN_ROOT="$TEMP/runs/resources"
mkdir -p "$MOCK_RUN_ROOT/preflight"
printf 'ok\n' > "$MOCK_RUN_ROOT/preflight/PREFLIGHT_OK"
bash "$SOURCE/server/launch_pipeline.sh" continue "$MOCK_RUN_ROOT" > "$TEMP/continue.out"
export MOCK_MEMORY_SUFFIX=G
bash "$SOURCE/server/retune_pending.sh" "$MOCK_RUN_ROOT" --apply > "$TEMP/retune.out"
[[ "$(wc -l < "$MOCK_RUN_ROOT/resource_overrides.tsv")" -eq 25 ]]
[[ "$(awk '$2==16384 {count++} END {print count+0}' "$MOCK_MEMORY")" -eq 22 ]]
[[ "$(awk '$2==32768 {count++} END {print count+0}' "$MOCK_MEMORY")" -eq 1 ]]
[[ "$(awk '$2==1024 {count++} END {print count+0}' "$MOCK_MEMORY")" -eq 1 ]]
[[ "$(wc -l < "$MOCK_LOG")" -eq 25 ]]
before="$(wc -l < "$MOCK_MEMORY")"
export MOCK_RUNNING_STAGE=train_C_1
if bash "$SOURCE/server/retune_pending.sh" "$MOCK_RUN_ROOT" --apply > "$TEMP/blocked.out" 2>&1; then
  echo 'Retune accepted a running job' >&2; exit 1
fi
[[ "$(wc -l < "$MOCK_MEMORY")" -eq "$before" ]]
unset MOCK_RUNNING_STAGE
export MOCK_FAIL_UPDATE_ON=16384
if bash "$SOURCE/server/retune_pending.sh" "$MOCK_RUN_ROOT" --apply > "$TEMP/update-failed.out" 2>&1; then
  echo 'Retune ignored a scheduler update failure' >&2; exit 1
fi
[[ "$(wc -l < "$MOCK_MEMORY")" -eq "$before" ]]
echo 'Pending job resource retune checks passed'

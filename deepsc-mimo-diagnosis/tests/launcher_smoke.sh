#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
TEMP="$(mktemp -d)"
trap 'rm -rf -- "$TEMP"' EXIT
mkdir -p "$TEMP/bin" "$TEMP/runs"
cp "$SOURCE/tests/mock_sbatch.sh" "$TEMP/bin/sbatch"
printf '#!/bin/bash\nexit 0\n' > "$TEMP/bin/flock"
chmod +x "$TEMP/bin/sbatch" "$TEMP/bin/flock"
export PATH="$TEMP/bin:$PATH"
export MOCK_COUNT="$TEMP/count" MOCK_LOG="$TEMP/sbatch.log"
export MOCK_SUFFIX=';mockcluster'
export DEEPSC_SHARED_ROOT="$TEMP/shared" DEEPSC_RUNS_ROOT="$TEMP/runs"
mkdir -p "$DEEPSC_SHARED_ROOT"

bash "$SOURCE/server/launch_pipeline.sh" preflight smoke > "$TEMP/preflight.out"
RUN_ROOT="$TEMP/runs/smoke"
mkdir -p "$RUN_ROOT/preflight"
printf 'ok\n' > "$RUN_ROOT/preflight/PREFLIGHT_OK"
bash "$SOURCE/server/launch_pipeline.sh" continue "$RUN_ROOT" > "$TEMP/continue.out"
[[ "$(wc -l < "$RUN_ROOT/jobs.tsv")" -eq 26 ]]
[[ "$(wc -l < "$MOCK_LOG")" -eq 25 ]]
[[ -f "$RUN_ROOT/PIPELINE_SUBMITTED" ]]
grep -q $'^train_C_1\t1002\tnone$' "$RUN_ROOT/jobs.tsv"
grep -q $'^train_C_2\t1003\t1002$' "$RUN_ROOT/jobs.tsv"
grep -q $'^integrity\t1018\t1009:1017$' "$RUN_ROOT/jobs.tsv"
grep -q $'^final_summary\t1025\t1024$' "$RUN_ROOT/jobs.tsv"
sed -n '2p' "$MOCK_LOG" | grep -q -- '--cpus-per-task=4'
sed -n '2p' "$MOCK_LOG" | grep -q -- '--mem=16G'
sed -n '19p' "$MOCK_LOG" | grep -q -- '--cpus-per-task=2'
sed -n '20p' "$MOCK_LOG" | grep -q -- '--cpus-per-task=4'
sed -n '20p' "$MOCK_LOG" | grep -q -- '--mem=32G'
sed -n '25p' "$MOCK_LOG" | grep -q -- '--cpus-per-task=1'
sed -n '25p' "$MOCK_LOG" | grep -q -- '--mem=1G'
bash "$SOURCE/server/launch_pipeline.sh" continue "$RUN_ROOT" > "$TEMP/repeat.out"
[[ "$(wc -l < "$MOCK_LOG")" -eq 25 ]]
cp -a "$RUN_ROOT" "$TEMP/runs/corrupt"
sed -i $'s/^train_C_2\t1003\t1002$/train_C_2\t1003\t9999/' "$TEMP/runs/corrupt/jobs.tsv"
if bash "$SOURCE/server/launch_pipeline.sh" continue "$TEMP/runs/corrupt" > "$TEMP/corrupt.out" 2>&1; then
  echo 'Incorrect recorded dependency was accepted' >&2; exit 1
fi
[[ "$(wc -l < "$MOCK_LOG")" -eq 25 ]]

export MOCK_FAIL_AT=1028
bash "$SOURCE/server/launch_pipeline.sh" preflight failed > "$TEMP/failed-preflight.out"
FAILED_ROOT="$TEMP/runs/failed"
mkdir -p "$FAILED_ROOT/preflight"
printf 'ok\n' > "$FAILED_ROOT/preflight/PREFLIGHT_OK"
if bash "$SOURCE/server/launch_pipeline.sh" continue "$FAILED_ROOT" > "$TEMP/failed.out" 2>&1; then
  echo 'Expected sbatch failure did not stop continuation' >&2; exit 1
fi
[[ "$(<"$FAILED_ROOT/SUBMISSION_IN_PROGRESS")" == train_C_2 ]]
[[ "$(wc -l < "$FAILED_ROOT/jobs.tsv")" -eq 3 ]]
if bash "$SOURCE/server/launch_pipeline.sh" continue "$FAILED_ROOT" > "$TEMP/blocked.out" 2>&1; then
  echo 'Uncertain submission was retried' >&2; exit 1
fi
[[ "$(wc -l < "$MOCK_LOG")" -eq 28 ]]
unset MOCK_FAIL_AT
# The mock failure is known not to have scheduled a job; a real operator must
# confirm that with the scheduler before removing this guard.
rm -- "$FAILED_ROOT/SUBMISSION_IN_PROGRESS"
bash "$SOURCE/server/launch_pipeline.sh" continue "$FAILED_ROOT" > "$TEMP/resumed.out"
[[ "$(wc -l < "$FAILED_ROOT/jobs.tsv")" -eq 26 ]]
[[ -f "$FAILED_ROOT/PIPELINE_SUBMITTED" ]]

export MOCK_BAD_AT=1053
bash "$SOURCE/server/launch_pipeline.sh" preflight invalid > "$TEMP/invalid-preflight.out"
INVALID_ROOT="$TEMP/runs/invalid"
mkdir -p "$INVALID_ROOT/preflight"
printf 'ok\n' > "$INVALID_ROOT/preflight/PREFLIGHT_OK"
if bash "$SOURCE/server/launch_pipeline.sh" continue "$INVALID_ROOT" > "$TEMP/invalid.out" 2>&1; then
  echo 'Invalid sbatch output was accepted' >&2; exit 1
fi
[[ -f "$INVALID_ROOT/SUBMISSION_IN_PROGRESS" ]]
echo 'Launcher mock submission, resume, and failure checks passed'

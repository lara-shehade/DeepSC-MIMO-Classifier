#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
TEMP="$(mktemp -d)"
trap 'rm -rf -- "$TEMP"' EXIT
RUN="$TEMP/run"
MOCK_LOG="$TEMP/scheduler.tsv"
export MOCK_LOG
mkdir -p "$RUN/code" "$RUN/checkpoints/C" "$RUN/checkpoints/D" \
  "$RUN/amendments/rng_resume_20260928" "$RUN/logs" \
  "$TEMP/shared/europarl"
cp "$SOURCE/main.py" "$RUN/code/main.py"
touch "$RUN/PIPELINE_SUBMITTED" "$RUN/amendments/rng_resume_20260928/record.tsv" \
  "$TEMP/shared/shared_vocab.json" "$TEMP/shared/europarl/train_data.pkl"
printf x > "$RUN/checkpoints/C/last.pth"
printf x > "$RUN/checkpoints/D/last.pth"
printf 'stage\tjob_id\tdependency\ntrain_C_1\t1001\tnone\ntrain_C_2\t1002\t1001\ntrain_C_3\t1003\t1002\ntrain_D_1\t2001\tnone\ntrain_D_2\t2002\t2001\ntrain_D_3\t2003\t2002\n' > "$RUN/jobs.tsv"
for model in C D; do
  printf '{"vocab_file": "%s/shared_vocab.json"}\n' "$TEMP/shared" \
    > "$RUN/checkpoints/$model/config.json"
done

sacct() {
  local id=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == -j ]]; then id="$2"; shift 2; else shift; fi
  done
  case "$id" in
    1001|2001) echo COMPLETED ;;
    1002|2002) echo FAILED ;;
    *) return 1 ;;
  esac
}
squeue() {
  local id="" name=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == -j ]]; then id="$2"; shift 2; else shift; fi
  done
  case "$id" in
    1003) name=ds_C_3 ;;
    2003) name=ds_D_3 ;;
    5001,1003|5002,2003) return 0 ;;
    *) return 1 ;;
  esac
  printf 'PENDING|%s\n' "$name"
}
sbatch() {
  local arg name=""
  for arg in "$@"; do
    case "$arg" in --job-name=*) name="${arg#--job-name=}" ;; esac
  done
  case "$name" in
    ds_C_2_recovery) echo 5001 ;;
    ds_D_2_recovery) echo 5002 ;;
    *) return 1 ;;
  esac
  printf '%s\n' "$name" >> "$MOCK_LOG"
}
scontrol() {
  if [[ "$1" == update ]]; then
    [[ "${2#JobId=}" != "${MOCK_FAIL_UPDATE_FOR:-}" ]] || return 42
    printf '%s\t%s\n' "${2#JobId=}" "${3#Dependency=}" >> "$MOCK_LOG"
  elif [[ "$1" == show ]]; then
    local id="$4" dependency
    dependency="$(awk -F '\t' -v id="$id" '$1==id {value=$2} END {print value}' "$MOCK_LOG")"
    printf 'JobId=%s JobState=PENDING Dependency=%s Reason=Dependency\n' "$id" "$dependency"
  else
    return 1
  fi
}
flock() { return 0; }
export -f sacct squeue sbatch scontrol flock

export MOCK_FAIL_UPDATE_FOR=1003
if bash "$SOURCE/server/recover_second_chunk.sh" "$RUN" C --apply > "$TEMP/c-update-failed.out" 2>&1; then
  echo 'Recovery ignored a failed dependency update' >&2
  exit 1
fi
unset MOCK_FAIL_UPDATE_FOR
bash "$SOURCE/server/recover_second_chunk.sh" "$RUN" C --apply > "$TEMP/c.out"
bash "$SOURCE/server/recover_second_chunk.sh" "$RUN" D --apply > "$TEMP/d.out"
bash "$SOURCE/server/recover_second_chunk.sh" "$RUN" C --apply > "$TEMP/c-repeat.out"
[[ "$(grep -c '^ds_[CD]_2_recovery$' "$MOCK_LOG")" -eq 2 ]]
[[ "$(<"$RUN/recovery/rng_resume_20260928/C_3_rewired_to")" == 5001 ]]
[[ "$(<"$RUN/recovery/rng_resume_20260928/D_3_rewired_to")" == 5002 ]]
echo 'Second-chunk recovery and no-duplicate checks passed'

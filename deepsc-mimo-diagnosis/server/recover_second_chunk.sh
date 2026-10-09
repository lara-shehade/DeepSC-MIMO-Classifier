#!/bin/bash
# Replace one failed chunk-2 job and rewire its still-pending chunk-3 successor.
set -euo pipefail

if [[ $# -ne 3 || "$1" != /* || "$3" != --apply || ! "$2" =~ ^[CD]$ ]]; then
  echo "Usage: bash $0 /absolute/path/to/run C|D --apply" >&2
  exit 2
fi
RUN_ROOT="$(cd "$1" && pwd)"
MODEL="$2"
SEED=42
[[ "$MODEL" == D ]] && SEED=43
PATCH_SHA=c64be3586cf51a25c0e04587d282b29edc4fb60e3065eff3d5de39706fea8617

[[ -f "$RUN_ROOT/PIPELINE_SUBMITTED" && -f "$RUN_ROOT/jobs.tsv" && \
   -f "$RUN_ROOT/amendments/rng_resume_20260928/record.tsv" ]] || {
  echo "Run is not ready for RNG-resume recovery" >&2; exit 2;
}
[[ "$(sha256sum "$RUN_ROOT/code/main.py" | awk '{print $1}')" == "$PATCH_SHA" ]] || {
  echo "RNG-resume fix is not installed in the run code" >&2; exit 2;
}

exec 9> "$RUN_ROOT/.rng_resume_recovery.lock"
flock -n 9 || { echo "Another recovery is in progress" >&2; exit 2; }
job_for() {
  awk -F '\t' -v stage="$1" '$1 == stage {print $2}' "$RUN_ROOT/jobs.tsv"
}
first_id="$(job_for "train_${MODEL}_1")"
failed_id="$(job_for "train_${MODEL}_2")"
successor_id="$(job_for "train_${MODEL}_3")"
for id in "$first_id" "$failed_id" "$successor_id"; do
  [[ "$id" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid job ledger" >&2; exit 2; }
done
state_of() {
  sacct -n -X -j "$1" -o State | awk 'NF {print $1; exit}'
}
[[ "$(state_of "$first_id")" == COMPLETED && \
   "$(state_of "$failed_id")" == FAILED && \
   -s "$RUN_ROOT/checkpoints/$MODEL/last.pth" ]] || {
  echo "Expected completed chunk 1, failed chunk 2, and intact checkpoint" >&2
  exit 2
}
status="$(squeue -h -j "$successor_id" -o '%T|%j')"
[[ "$status" == "PENDING|ds_${MODEL}_3" ]] || {
  echo "Chunk 3 is not the expected pending job: $status" >&2; exit 2;
}

vocab_path="$(awk -F '"' '$2 == "vocab_file" {print $4; exit}' \
  "$RUN_ROOT/checkpoints/$MODEL/config.json")"
[[ "$vocab_path" == */shared_vocab.json && -f "$vocab_path" ]] || {
  echo "Cannot determine the approved shared data root from checkpoint config" >&2
  exit 2
}
shared_root="${vocab_path%/shared_vocab.json}"
[[ -f "$shared_root/europarl/train_data.pkl" ]] || {
  echo "Europarl training pickle is missing under $shared_root" >&2; exit 2;
}

recovery_dir="$RUN_ROOT/recovery/rng_resume_20260928"
mkdir -p "$recovery_dir"
id_file="$recovery_dir/${MODEL}_2_replacement_job_id"
guard_file="$recovery_dir/${MODEL}_2_SUBMISSION_IN_PROGRESS"
if [[ -f "$id_file" ]]; then
  replacement_id="$(<"$id_file")"
  [[ "$replacement_id" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid recorded replacement job ID" >&2; exit 2;
  }
else
  [[ ! -e "$guard_file" ]] || {
    echo "Prior submission outcome is uncertain; inspect $guard_file" >&2; exit 2;
  }
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$guard_file"
  if output="$(sbatch --parsable \
      --partition="${DEEPSC_GPU_PARTITION:-gpu}" \
      --gres="${DEEPSC_GPU_RESOURCE:-gpu:v100d32q:1}" \
      --cpus-per-task=4 --mem=16G --time=05:50:00 \
      --job-name="ds_${MODEL}_2_recovery" \
      --output="$RUN_ROOT/logs/train-${MODEL}-2-recovery-%j.out" \
      --error="$RUN_ROOT/logs/train-${MODEL}-2-recovery-%j.err" \
      --export="ALL,PROJECT_ROOT=$RUN_ROOT/code,RUN_ROOT=$RUN_ROOT,DEEPSC_SHARED_ROOT=$shared_root,MODEL_ID=$MODEL,MODEL_SEED=$SEED,CHUNK_INDEX=2" \
      "$RUN_ROOT/code/server/train_chunk.slurm")"; then :; else
    echo "Replacement chunk 2 submission failed; inspect $guard_file" >&2
    exit 1
  fi
  [[ "$output" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
    echo "Unexpected sbatch response: $output; inspect $guard_file" >&2; exit 1;
  }
  replacement_id="${BASH_REMATCH[1]}"
  printf '%s\n' "$replacement_id" > "$id_file"
  rm -f -- "$guard_file"
fi

scontrol update JobId="$successor_id" Dependency="afterok:$replacement_id"
details="$(scontrol show job -o "$successor_id")"
[[ "$details" == *"Dependency=afterok:$replacement_id "* || \
   "$details" == *"Dependency=afterok:$replacement_id" ]] || {
  echo "Replacement $replacement_id submitted, but chunk 3 dependency was not verified" >&2
  exit 1
}
printf '%s\n' "$replacement_id" > "$recovery_dir/${MODEL}_3_rewired_to"
echo "${MODEL} recovery: replacement chunk 2 = $replacement_id; chunk 3 = $successor_id rewired"
squeue -h -j "$replacement_id,$successor_id" -o '%i %j %T %R'

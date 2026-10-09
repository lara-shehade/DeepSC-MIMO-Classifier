#!/bin/bash
# Recover only a timed-out final frozen classical fit.
set -euo pipefail

[[ $# -eq 2 && ( "$1" == check || "$1" == start ) ]] || {
  echo "Usage: bash $0 check|start /absolute/run/root" >&2; exit 2;
}
MODE="$1"
RUN_ROOT="$(cd "$2" && pwd)"
PROJECT_ROOT="$RUN_ROOT/code"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FREEZE_ROOT="$RUN_ROOT/classical_freeze_recovery_v1"
PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"
OUTPUT_ROOT="$RUN_ROOT/analysis/classical"

[[ -s "$RUN_ROOT/analysis/development_features.csv" ]] || {
  echo "Missing feature CSV" >&2; exit 2;
}
mkdir -p "$FREEZE_ROOT/logs"
exec 9> "$FREEZE_ROOT/.submission.lock"
flock -n 9 || { echo "Another freeze recovery launcher is active" >&2; exit 2; }

if [[ ! -d "$FREEZE_ROOT/code/classical_freeze_recovery" ]]; then
  mkdir -p "$FREEZE_ROOT/code"
  cp -a "$SOURCE_ROOT/classical_freeze_recovery" "$FREEZE_ROOT/code/"
  (cd "$FREEZE_ROOT" && find code/classical_freeze_recovery -maxdepth 1 -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) -print0 | \
    sort -z | xargs -0 sha256sum > code.sha256)
  (cd "$PROJECT_ROOT" && sha256sum analysis/classical_diagnostics.py \
    > "$FREEZE_ROOT/dependency.sha256")
fi
(cd "$FREEZE_ROOT" && sha256sum -c code.sha256 >/dev/null)
(cd "$PROJECT_ROOT" && sha256sum -c "$FREEZE_ROOT/dependency.sha256" >/dev/null)
for script in "$FREEZE_ROOT/code/classical_freeze_recovery/"*.sh \
              "$FREEZE_ROOT/code/classical_freeze_recovery/"*.slurm; do
  bash -n "$script"
done
module purge
module load python/ai-4
cd "$PROJECT_ROOT"
export PYTHONPATH="$FREEZE_ROOT/code:$PROJECT_ROOT:${PYTHONPATH:-}"
python -m py_compile "$FREEZE_ROOT/code/classical_freeze_recovery/"*.py

OLD_FINAL="$(awk -F '\t' '$1=="finalize"{print $2}' \
  "$RUN_ROOT/classical_tail_v1/jobs.tsv")"
HOLDOUT="$(awk -F '\t' '$1=="collect_arxiv_holdout"{print $2}' "$RUN_ROOT/jobs.tsv")"
SUBSET_PILOT="$(awk -F '\t' '$1=="pilot"{print $2}' "$RUN_ROOT/stage_subset_v1/jobs.tsv")"
for value in "$OLD_FINAL" "$HOLDOUT" "$SUBSET_PILOT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid recorded job ID" >&2; exit 2; }
done

if [[ -s "$OUTPUT_ROOT/FROZEN" ]]; then
  echo "Classical finalization already completed; no recovery needed"
  exit 0
fi
python -m classical_freeze_recovery.recover check --output-root "$OUTPUT_ROOT"
STATE="$(sacct -X -n -P -j "$OLD_FINAL" -o State | awk -F '|' 'NF{print $1;exit}')"
if [[ "$MODE" == check ]]; then
  echo "Current finalizer state: ${STATE:-unknown}"
  echo "Freeze recovery check passed; no jobs submitted"
  exit 0
fi
[[ "$STATE" == TIMEOUT || "$STATE" == FAILED || "$STATE" == CANCELLED ]] || {
  echo "Finalizer $OLD_FINAL has not failed (state ${STATE:-unknown}); do not recover yet" >&2
  exit 2
}
[[ ! -e "$FREEZE_ROOT/jobs.tsv" && ! -e "$FREEZE_ROOT/SUBMISSION_IN_PROGRESS" ]] || {
  echo "Freeze recovery already submitted or submission is uncertain" >&2; exit 2;
}
for job in "$HOLDOUT" "$SUBSET_PILOT"; do
  [[ "$(squeue -h -j "$job" -o '%T')" == PENDING ]] || {
    echo "Expected pending downstream job $job" >&2; exit 2;
  }
done

printf 'stage\tjob_id\tdependency\n' > "$FREEZE_ROOT/jobs.tsv"
EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT,FREEZE_ROOT=$FREEZE_ROOT"
submit() {
  local stage="$1" dependency="$2" chunk="$3" output status job_id
  printf '%s\n' "$stage" > "$FREEZE_ROOT/SUBMISSION_IN_PROGRESS"
  if [[ "$dependency" == none ]]; then
    if output="$(sbatch --parsable --partition="$PARTITION" --time=06:00:00 \
      --cpus-per-task=2 --mem=16G --job-name="ds_freeze_${chunk}" \
      --output="$FREEZE_ROOT/logs/chunk-${chunk}-%j.out" \
      --error="$FREEZE_ROOT/logs/chunk-${chunk}-%j.err" \
      --export="$EXPORTS,FREEZE_CHUNK=$chunk" \
      "$FREEZE_ROOT/code/classical_freeze_recovery/chunk.slurm")"; then :; else
      status=$?; echo "sbatch failed for $stage (exit $status)" >&2; return "$status"
    fi
  else
    if output="$(sbatch --parsable --partition="$PARTITION" --time=06:00:00 \
      --dependency="afterok:$dependency" --cpus-per-task=2 --mem=16G \
      --job-name="ds_freeze_${chunk}" \
      --output="$FREEZE_ROOT/logs/chunk-${chunk}-%j.out" \
      --error="$FREEZE_ROOT/logs/chunk-${chunk}-%j.err" \
      --export="$EXPORTS,FREEZE_CHUNK=$chunk" \
      "$FREEZE_ROOT/code/classical_freeze_recovery/chunk.slurm")"; then :; else
      status=$?; echo "sbatch failed for $stage (exit $status)" >&2; return "$status"
    fi
  fi
  [[ "$output" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
    echo "Invalid sbatch response: $output" >&2; return 1;
  }
  job_id="${BASH_REMATCH[1]}"
  printf '%s\t%s\t%s\n' "$stage" "$job_id" "$dependency" >> "$FREEZE_ROOT/jobs.tsv"
  LAST_JOB="$job_id"
  rm -f "$FREEZE_ROOT/SUBMISSION_IN_PROGRESS"
}

submit chunk_1 none 0
CHUNK1="$LAST_JOB"
submit chunk_2 "$CHUNK1" 1
CHUNK2="$LAST_JOB"
scontrol update JobId="$HOLDOUT" Dependency="afterok:$CHUNK2"
scontrol update JobId="$SUBSET_PILOT" Dependency="afterok:$CHUNK2"
for job in "$HOLDOUT" "$SUBSET_PILOT"; do
  record="$(scontrol show job -o "$job")"
  [[ "$record" == *"Dependency=afterok:$CHUNK2"* ]] || {
    echo "Could not verify dependency repair for $job" >&2; exit 1;
  }
done
printf 'holdout\t%s\tafterok:%s\nsubset_pilot\t%s\tafterok:%s\n' \
  "$HOLDOUT" "$CHUNK2" "$SUBSET_PILOT" "$CHUNK2" >> "$FREEZE_ROOT/jobs.tsv"
touch "$FREEZE_ROOT/SUBMITTED"
echo "Submitted two resumable freeze chunks: $CHUNK1 then $CHUNK2"

#!/bin/bash
# Recover only timed-out fusion folds 22--24, then reconnect blocked DAGs.
set -euo pipefail

[[ $# -eq 2 && ( "$1" == check || "$1" == start ) ]] || {
  echo "Usage: bash $0 check|start /absolute/run/root" >&2; exit 2;
}
MODE="$1"
RUN_ROOT="$(cd "$2" && pwd)"
PROJECT_ROOT="$RUN_ROOT/code"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECOVERY_ROOT="$RUN_ROOT/classical_tail_v1"
GPU_PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"

[[ -s "$RUN_ROOT/analysis/development_features.csv" ]] || {
  echo "Missing completed feature CSV" >&2; exit 2;
}
[[ -f "$PROJECT_ROOT/server/classical_recovery.py" ]] || {
  echo "Existing recovery code is missing from run snapshot" >&2; exit 2;
}
mkdir -p "$RECOVERY_ROOT/logs"
exec 9> "$RECOVERY_ROOT/.submission.lock"
flock -n 9 || { echo "Another tail recovery launcher is active" >&2; exit 2; }

if [[ ! -d "$RECOVERY_ROOT/code/classical_tail" ]]; then
  mkdir -p "$RECOVERY_ROOT/code"
  cp -a "$SOURCE_ROOT/classical_tail" "$RECOVERY_ROOT/code/classical_tail"
  (
    cd "$RECOVERY_ROOT"
    find code/classical_tail -maxdepth 1 -type f \
      \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) \
      -print0 | sort -z | xargs -0 sha256sum > code.sha256
  )
  (
    cd "$PROJECT_ROOT"
    sha256sum analysis/classical_diagnostics.py server/classical_recovery.py \
      > "$RECOVERY_ROOT/dependency.sha256"
  )
fi
(cd "$RECOVERY_ROOT" && sha256sum -c code.sha256 >/dev/null) || {
  echo "Recovery snapshot changed" >&2; exit 2;
}
(cd "$PROJECT_ROOT" && sha256sum -c "$RECOVERY_ROOT/dependency.sha256" >/dev/null) || {
  echo "Classical source changed" >&2; exit 2;
}
module purge
module load python/ai-4
cd "$PROJECT_ROOT"
export PYTHONPATH="$RECOVERY_ROOT/code:$PROJECT_ROOT:${PYTHONPATH:-}"
python -m py_compile "$RECOVERY_ROOT/code/classical_tail/"*.py
python -m classical_tail.recover check --output-root "$RUN_ROOT/analysis/classical"

OLD_ARRAY="$(awk -F '\t' '$1=="folds"{print $2}' "$RUN_ROOT/classical_recovery_jobs.tsv")"
OLD_FINAL="$(awk -F '\t' '$1=="finalize"{print $2}' "$RUN_ROOT/classical_recovery_jobs.tsv")"
HOLDOUT="$(awk -F '\t' '$1=="collect_arxiv_holdout"{print $2}' "$RUN_ROOT/jobs.tsv")"
SUBSET_PILOT="$(awk -F '\t' '$1=="pilot"{print $2}' "$RUN_ROOT/stage_subset_v1/jobs.tsv")"
for VALUE in "$OLD_ARRAY" "$OLD_FINAL" "$HOLDOUT" "$SUBSET_PILOT"; do
  [[ "$VALUE" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid recorded job ID" >&2; exit 2; }
done
OLD_ARRAY_STATES="$(sacct -X -n -P -j "$OLD_ARRAY" -o State)"
[[ "$OLD_ARRAY_STATES" == *TIMEOUT* || "$OLD_ARRAY_STATES" == *FAILED* ]] || {
  echo "Old fold array contains no failed/timed-out task" >&2; exit 2;
}
for JOB in "$OLD_FINAL" "$HOLDOUT" "$SUBSET_PILOT"; do
  [[ "$(squeue -h -j "$JOB" -o '%T')" == PENDING ]] || {
    echo "Expected pending downstream job $JOB" >&2; exit 2;
  }
done
if [[ "$MODE" == check ]]; then
  echo "Tail recovery check passed; no jobs submitted"
  exit 0
fi
[[ ! -e "$RECOVERY_ROOT/jobs.tsv" && ! -e "$RECOVERY_ROOT/SUBMISSION_IN_PROGRESS" ]] || {
  echo "Tail recovery was already submitted or submission is uncertain" >&2; exit 2;
}
printf 'stage\tjob_id\tdependency\n' > "$RECOVERY_ROOT/jobs.tsv"
EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT,RECOVERY_ROOT=$RECOVERY_ROOT"

submit() {
  local stage="$1" dependency="$2" output job_id status
  shift 2
  printf '%s\n' "$stage" > "$RECOVERY_ROOT/SUBMISSION_IN_PROGRESS"
  if [[ "$dependency" == none ]]; then
    if output="$(sbatch --parsable --partition="$GPU_PARTITION" "$@")"; then :; else
      status=$?; echo "sbatch failed for $stage (exit $status)" >&2; return "$status"
    fi
  else
    if output="$(sbatch --parsable --partition="$GPU_PARTITION" \
      --dependency="afterok:$dependency" "$@")"; then :; else
      status=$?; echo "sbatch failed for $stage (exit $status)" >&2; return "$status"
    fi
  fi
  [[ "$output" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
    echo "Invalid sbatch response for $stage: $output" >&2; return 1;
  }
  job_id="${BASH_REMATCH[1]}"
  printf '%s\t%s\t%s\n' "$stage" "$job_id" "$dependency" >> "$RECOVERY_ROOT/jobs.tsv"
  LAST_JOB="$job_id"
  rm -f "$RECOVERY_ROOT/SUBMISSION_IN_PROGRESS"
}

submit pilot none --cpus-per-task=2 --mem=16G --job-name=ds_ctail_pilot \
  --output="$RECOVERY_ROOT/logs/pilot-%j.out" \
  --error="$RECOVERY_ROOT/logs/pilot-%j.err" \
  --export="$EXPORTS,TAIL_PILOT=1" \
  "$RECOVERY_ROOT/code/classical_tail/part.slurm"
PILOT="$LAST_JOB"
submit remaining "$PILOT" --array='1-8%2' --cpus-per-task=2 --mem=16G \
  --job-name=ds_ctail_parts \
  --output="$RECOVERY_ROOT/logs/part-%A_%a.out" \
  --error="$RECOVERY_ROOT/logs/part-%A_%a.err" \
  --export="$EXPORTS,TAIL_PILOT=0" "$RECOVERY_ROOT/code/classical_tail/part.slurm"
PARTS="$LAST_JOB"
submit finalize "$PARTS" --cpus-per-task=2 --mem=16G --job-name=ds_ctail_final \
  --output="$RECOVERY_ROOT/logs/final-%j.out" \
  --error="$RECOVERY_ROOT/logs/final-%j.err" \
  --export="$EXPORTS" "$RECOVERY_ROOT/code/classical_tail/finalize.slurm"
FINAL="$LAST_JOB"

scontrol update JobId="$HOLDOUT" Dependency="afterok:$FINAL"
scontrol update JobId="$SUBSET_PILOT" Dependency="afterok:$FINAL"
for JOB in "$HOLDOUT" "$SUBSET_PILOT"; do
  RECORD="$(scontrol show job -o "$JOB")"
  [[ "$RECORD" == *"Dependency=afterok:$FINAL"* ]] || {
    echo "Could not verify repaired dependency for $JOB" >&2; exit 1;
  }
done
printf 'holdout\t%s\tafterok:%s\nsubset_pilot\t%s\tafterok:%s\n' \
  "$HOLDOUT" "$FINAL" "$SUBSET_PILOT" "$FINAL" >> "$RECOVERY_ROOT/jobs.tsv"
scancel "$OLD_FINAL"
touch "$RECOVERY_ROOT/SUBMITTED"
echo "Submitted missing estimator parts; finalizer $FINAL now gates arXiv $HOLDOUT and subset pilot $SUBSET_PILOT"

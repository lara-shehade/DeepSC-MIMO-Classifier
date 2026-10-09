#!/bin/bash
# Check, pilot, then submit the remaining post-hoc subset folds.
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "Usage: $0 check|pilot|continue /absolute/run/root" >&2
  exit 2
}
MODE="$1"
[[ "$MODE" == check || "$MODE" == pilot || "$MODE" == continue ]] || {
  echo "Invalid mode: $MODE" >&2; exit 2;
}
RUN_ROOT="$(cd "$2" && pwd)"
PROJECT_ROOT="$RUN_ROOT/code"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_ROOT="$RUN_ROOT/stage_subset_v1"
[[ -f "$PROJECT_ROOT/analysis/neural_diagnostics.py" ]] || {
  echo "Missing original neural analysis under $PROJECT_ROOT" >&2; exit 2;
}

mkdir -p "$STUDY_ROOT/logs"
exec 9> "$STUDY_ROOT/.submission.lock"
flock -n 9 || { echo "Another subset submission is active" >&2; exit 2; }
if [[ ! -d "$STUDY_ROOT/code/stage_subset" ]]; then
  [[ ! -e "$STUDY_ROOT/code.sha256" ]] || {
    echo "Incomplete source snapshot; inspect $STUDY_ROOT" >&2; exit 2;
  }
  mkdir -p "$STUDY_ROOT/code"
  cp -a "$SOURCE_ROOT/stage_subset" "$STUDY_ROOT/code/stage_subset"
  (
    cd "$STUDY_ROOT"
    find code/stage_subset -maxdepth 1 -type f \
      \( -name '*.py' -o -name '*.slurm' -o -name '*.sh' -o -name README.md \) \
      -print0 | sort -z | xargs -0 sha256sum > code.sha256
    tar -czf source_snapshot.tar.gz code
  )
  sha256sum "$PROJECT_ROOT/analysis/neural_diagnostics.py" > "$STUDY_ROOT/dependency.sha256"
fi
[[ -s "$STUDY_ROOT/code.sha256" && -s "$STUDY_ROOT/dependency.sha256" ]] || {
  echo "Missing subset or dependency source hashes" >&2; exit 2;
}
(cd "$STUDY_ROOT" && sha256sum -c code.sha256 >/dev/null) || {
  echo "Subset source snapshot changed; refusing submission" >&2; exit 2;
}
sha256sum -c "$STUDY_ROOT/dependency.sha256" >/dev/null || {
  echo "Original neural analysis changed since subset snapshot" >&2; exit 2;
}
[[ -d "$RUN_ROOT/traces/development" ]] || {
  echo "Missing saved development traces" >&2; exit 2;
}
[[ -n "$(find "$RUN_ROOT/traces/development" -type f -name '*.pt' -print -quit)" ]] || {
  echo "No development artifacts found" >&2; exit 2;
}
for STAGE in R0 R1 R2 R3; do
  for SEED in 3101 3102 3103 3104 3105; do
    [[ -s "$RUN_ROOT/analysis/neural_folds/tcn_${STAGE}_seed${SEED}.json" ]] || {
      echo "Missing existing TCN fold: $STAGE seed $SEED" >&2; exit 2;
    }
  done
done

validate_existing() {
  cd "$PROJECT_ROOT"
  PYTHONPATH="$STUDY_ROOT/code:$PROJECT_ROOT" python -m stage_subset.validate_existing \
    "$RUN_ROOT/analysis/neural_folds"
}

run_smoke() {
  module purge
  module load python/ai-4
  cd "$PROJECT_ROOT"
  if PYTHONPATH="$STUDY_ROOT/code:$PROJECT_ROOT" \
    python -m stage_subset.smoke > "$STUDY_ROOT/logs/smoke.log" 2>&1; then
    echo "Subset smoke test passed"
  else
    tail -n 15 "$STUDY_ROOT/logs/smoke.log" >&2
    echo "No new jobs submitted" >&2
    return 1
  fi
}

submit_job() {
  local output
  if output="$(sbatch --parsable "$@")"; then :; else
    echo "sbatch failed; inspect submission guard and jobs.tsv" >&2
    return 1
  fi
  if [[ ! "$output" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]]; then
    echo "Invalid sbatch response: $output" >&2
    return 1
  fi
  printf '%s' "${BASH_REMATCH[1]}"
}

LEDGER="$STUDY_ROOT/jobs.tsv"
GPU_PARTITION="${DEEPSC_GPU_PARTITION:-gpu}"
CPU_PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"
GPU_RESOURCE="${DEEPSC_GPU_RESOURCE:-gpu:v100d32q:1}"
EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT,STUDY_ROOT=$STUDY_ROOT"

if [[ "$MODE" == check ]]; then
  module purge
  module load python/ai-4
  validate_existing
  run_smoke
  echo "Check passed; no jobs submitted"
  exit 0
fi

[[ -s "$RUN_ROOT/analysis/classical/grouped_results.json" && \
   -s "$RUN_ROOT/analysis/classical/FROZEN" && \
   -s "$RUN_ROOT/analysis/frozen_neural.selection.json" && \
   -s "$RUN_ROOT/analysis/frozen_neural.FROZEN" ]] || {
  echo "Classical and neural development results are not both frozen; wait for them. arXiv is not required." >&2
  exit 2
}

if [[ "$MODE" == pilot ]]; then
  [[ ! -e "$LEDGER" && ! -e "$STUDY_ROOT/PILOT_SUBMISSION_IN_PROGRESS" ]] || {
    echo "Pilot already submitted or prior submission uncertain; inspect $STUDY_ROOT" >&2
    exit 2
  }
  module purge
  module load python/ai-4
  validate_existing
  run_smoke
  printf 'pilot\n' > "$STUDY_ROOT/PILOT_SUBMISSION_IN_PROGRESS"
  if PILOT_JOB="$(submit_job --partition="$GPU_PARTITION" --gres="$GPU_RESOURCE" \
    --cpus-per-task=2 --mem=16G --job-name=ds_subpilot \
    --output="$STUDY_ROOT/logs/pilot-%j.out" \
    --error="$STUDY_ROOT/logs/pilot-%j.err" \
    --export="$EXPORTS,TASK_INDEX=50" \
    "$STUDY_ROOT/code/stage_subset/fold.slurm")"; then :; else exit 1; fi
  printf 'stage\tjob_id\tdependency\npilot\t%s\tnone\n' "$PILOT_JOB" > "$LEDGER"
  mv "$STUDY_ROOT/PILOT_SUBMISSION_IN_PROGRESS" "$STUDY_ROOT/PILOT_SUBMITTED"
  printf 'Submitted full-data four-stage pilot %s\n' "$PILOT_JOB"
  exit 0
fi

[[ -f "$LEDGER" && -f "$STUDY_ROOT/PILOT_SUBMITTED" ]] || {
  echo "Pilot submission record missing" >&2; exit 2;
}
[[ ! -e "$STUDY_ROOT/ARRAY_SUBMISSION_IN_PROGRESS" && \
   ! -e "$STUDY_ROOT/ARRAY_SUBMITTED" ]] || {
  echo "Array was already submitted or prior submission is uncertain" >&2; exit 2;
}
PILOT_JOB="$(awk -F '\t' '$1=="pilot"{print $2}' "$LEDGER")"
[[ "$PILOT_JOB" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid pilot job ID in $LEDGER" >&2; exit 2;
}
PILOT_ACCOUNTING="$(sacct -X -n -P -j "$PILOT_JOB" -o State,ExitCode)"
[[ "$PILOT_ACCOUNTING" == 'COMPLETED|0:0' ]] || {
  echo "Pilot has not completed successfully: $PILOT_ACCOUNTING" >&2; exit 2;
}
PILOT_OUTPUT="$STUDY_ROOT/results/folds/R0_R1_R2_R3_seed3101.json"
[[ -s "$PILOT_OUTPUT" ]] || { echo "Pilot output is missing" >&2; exit 2; }
PYTHONPATH="$STUDY_ROOT/code:$PROJECT_ROOT" python -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["task_index"]==50 and r["held_out_physical_seed"]==3101 and r["stages"]==["R0","R1","R2","R3"] and r["training"]["epochs"]==12 and 0<=r["macro_f1"]<=1' \
  "$PILOT_OUTPUT" || { echo "Pilot report is invalid" >&2; exit 2; }

printf 'remaining folds\n' > "$STUDY_ROOT/ARRAY_SUBMISSION_IN_PROGRESS"
if ARRAY_JOB="$(submit_job --partition="$GPU_PARTITION" --gres="$GPU_RESOURCE" \
  --array='0-49,51-54%4' --cpus-per-task=2 --mem=16G \
  --job-name=ds_subsets \
  --output="$STUDY_ROOT/logs/fold-%A_%a.out" \
  --error="$STUDY_ROOT/logs/fold-%A_%a.err" \
  --export="$EXPORTS" \
  "$STUDY_ROOT/code/stage_subset/fold.slurm")"; then :; else exit 1; fi
printf 'folds\t%s\tnone\n' "$ARRAY_JOB" >> "$LEDGER"
if SUMMARY_JOB="$(submit_job --partition="$CPU_PARTITION" \
  --dependency="afterok:$ARRAY_JOB" --cpus-per-task=1 --mem=2G \
  --job-name=ds_subsummary \
  --output="$STUDY_ROOT/logs/summary-%j.out" \
  --error="$STUDY_ROOT/logs/summary-%j.err" \
  --export="$EXPORTS" \
  "$STUDY_ROOT/code/stage_subset/summary.slurm")"; then :; else
  echo "Folds submitted as $ARRAY_JOB, but summary submission failed" >&2
  exit 1
fi
printf 'summary\t%s\tafterok:%s\n' "$SUMMARY_JOB" "$ARRAY_JOB" >> "$LEDGER"
mv "$STUDY_ROOT/ARRAY_SUBMISSION_IN_PROGRESS" "$STUDY_ROOT/ARRAY_SUBMITTED"
printf 'Submitted remaining folds %s and summary %s\n' "$ARRAY_JOB" "$SUMMARY_JOB"

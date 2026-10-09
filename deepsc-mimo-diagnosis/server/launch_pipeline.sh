#!/bin/bash
set -euo pipefail

usage() {
  printf 'Usage: %s preflight [RUN_ID] | continue /absolute/path/to/run\n' "$0" >&2
}

MODE="${1:-}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_PARTITION="${DEEPSC_GPU_PARTITION:-gpu}"
CPU_PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"
GPU_RESOURCE="${DEEPSC_GPU_RESOURCE:-gpu:v100d32q:1}"

EXPECTED_STAGES=(preflight)
for MODEL in C D; do
  for CHUNK in {1..8}; do EXPECTED_STAGES+=("train_${MODEL}_${CHUNK}"); done
done
EXPECTED_STAGES+=(integrity collect_development classical neural_folds neural_freeze
  collect_arxiv_holdout evaluate_arxiv_holdout final_summary)
declare -A JOB_IDS=() JOB_DEPS=() SEEN_IDS=()
LAST_JOB_ID=""

load_ledger() {
  local header stage job_id dependency extra index=0
  [[ -f "$RUN_ROOT/jobs.tsv" ]] || { echo "Missing jobs.tsv" >&2; return 1; }
  IFS= read -r header < "$RUN_ROOT/jobs.tsv"
  [[ "$header" == $'stage\tjob_id\tdependency' ]] || {
    echo "Invalid jobs.tsv header" >&2; return 1;
  }
  while IFS=$'\t' read -r stage job_id dependency extra; do
    [[ "$stage" == "${EXPECTED_STAGES[$index]-}" && -z "${extra:-}" ]] || {
      echo "Unexpected, duplicate, or out-of-order stage in jobs.tsv: $stage" >&2; return 1;
    }
    [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || {
      echo "Invalid job ID for $stage in jobs.tsv: $job_id" >&2; return 1;
    }
    [[ -z "${SEEN_IDS[$job_id]-}" ]] || {
      echo "Duplicate job ID in jobs.tsv: $job_id" >&2; return 1;
    }
    [[ "$dependency" == none || "$dependency" =~ ^[1-9][0-9]*(:[1-9][0-9]*)*$ ]] || {
      echo "Invalid dependency for $stage in jobs.tsv: $dependency" >&2; return 1;
    }
    JOB_IDS["$stage"]="$job_id"
    JOB_DEPS["$stage"]="$dependency"
    SEEN_IDS["$job_id"]=1
    index=$((index + 1))
  done < <(tail -n +2 "$RUN_ROOT/jobs.tsv")
}

submit_stage() {
  local stage="$1" partition="$2" dependency="$3" output job_id status
  shift 3
  if [[ -n "${JOB_IDS[$stage]-}" ]]; then
    [[ "${JOB_DEPS[$stage]}" == "$dependency" ]] || {
      echo "Recorded dependency mismatch for $stage" >&2; return 1;
    }
    LAST_JOB_ID="${JOB_IDS[$stage]}"
    return 0
  fi
  # A crash between sbatch and the ledger append has an uncertain outcome.
  # Refuse automatic retry in that case to avoid submitting the same job twice.
  printf '%s\n' "$stage" > "$RUN_ROOT/SUBMISSION_IN_PROGRESS" || {
    echo "Cannot write submission guard for $stage" >&2; return 1;
  }
  if [[ "$dependency" == none ]]; then
    if output="$(sbatch --parsable --partition="$partition" "$@")"; then :; else
      status=$?
      echo "sbatch failed for $stage (exit $status); inspect SUBMISSION_IN_PROGRESS before retrying" >&2
      return "$status"
    fi
  else
    if output="$(sbatch --parsable --partition="$partition" \
      --dependency="afterok:$dependency" "$@")"; then :; else
      status=$?
      echo "sbatch failed for $stage (exit $status); inspect SUBMISSION_IN_PROGRESS before retrying" >&2
      return "$status"
    fi
  fi
  if [[ ! "$output" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]]; then
    echo "Invalid sbatch --parsable response for $stage: $output" >&2
    return 1
  fi
  job_id="${BASH_REMATCH[1]}"
  [[ -z "${SEEN_IDS[$job_id]-}" ]] || {
    echo "sbatch returned duplicate job ID $job_id for $stage" >&2; return 1;
  }
  printf '%s\t%s\t%s\n' "$stage" "$job_id" "$dependency" >> "$RUN_ROOT/jobs.tsv" || {
    echo "Submitted $stage as $job_id but could not write jobs.tsv; inspect the scheduler" >&2
    return 1
  }
  JOB_IDS["$stage"]="$job_id"
  JOB_DEPS["$stage"]="$dependency"
  SEEN_IDS["$job_id"]=1
  LAST_JOB_ID="$job_id"
  rm -f -- "$RUN_ROOT/SUBMISSION_IN_PROGRESS" || {
    echo "Could not clear submission guard for $stage" >&2; return 1;
  }
}

if [[ "$MODE" == preflight ]]; then
  [[ $# -le 2 ]] || { usage; exit 2; }
  : "${DEEPSC_SHARED_ROOT:?Set the shared adv_datasets directory}"
  : "${DEEPSC_RUNS_ROOT:?Set a scratch directory for immutable runs}"
  RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
  [[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid run ID" >&2; exit 2; }
  RUN_ROOT="${DEEPSC_RUNS_ROOT%/}/${RUN_ID}"
  if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to reuse existing run: $RUN_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RUN_ROOT/logs"
  cp -a "$SOURCE_ROOT" "$RUN_ROOT/code"
  PROJECT_ROOT="$RUN_ROOT/code"
  tar -czf "$RUN_ROOT/source_snapshot.tar.gz" -C "$RUN_ROOT" code
  sha256sum "$RUN_ROOT/source_snapshot.tar.gz" > "$RUN_ROOT/source_snapshot.sha256"
  printf 'stage\tjob_id\tdependency\n' > "$RUN_ROOT/jobs.tsv"
  EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT,DEEPSC_SHARED_ROOT=$DEEPSC_SHARED_ROOT"
  submit_stage preflight "$GPU_PARTITION" none --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G \
    --job-name="ds_preflight_${RUN_ID}" \
    --output="$RUN_ROOT/logs/preflight-%j.out" \
    --error="$RUN_ROOT/logs/preflight-%j.err" \
    --export="$EXPORTS" "$PROJECT_ROOT/server/preflight_full.slurm" || exit 1
  printf '%s\n' "$RUN_ROOT" > "$RUN_ROOT/RUN_ROOT"
  printf 'Submitted preflight job %s\n' "$LAST_JOB_ID"
  printf 'After it succeeds, run: %s/server/launch_pipeline.sh continue %s\n' "$PROJECT_ROOT" "$RUN_ROOT"
  exit 0
fi

if [[ "$MODE" != continue || $# -ne 2 ]]; then usage; exit 2; fi
RUN_ROOT="$(cd "$2" && pwd)"
PROJECT_ROOT="$RUN_ROOT/code"
DEEPSC_SHARED_ROOT="${DEEPSC_SHARED_ROOT:?Export the same shared adv_datasets directory}"
[[ -f "$RUN_ROOT/preflight/PREFLIGHT_OK" ]] || {
  echo "Preflight has not passed for $RUN_ROOT" >&2; exit 2;
}
[[ -f "$RUN_ROOT/source_snapshot.sha256" ]] || {
  echo "Missing source snapshot hash" >&2; exit 2;
}
(cd "$RUN_ROOT" && sha256sum -c source_snapshot.sha256) || exit 1
exec 9> "$RUN_ROOT/.launch.lock"
flock -n 9 || { echo "Another continuation is running for $RUN_ROOT" >&2; exit 2; }
load_ledger || exit 1
[[ -n "${JOB_IDS[preflight]-}" ]] || { echo "Preflight job is missing from jobs.tsv" >&2; exit 2; }
if [[ -e "$RUN_ROOT/SUBMISSION_IN_PROGRESS" ]]; then
  echo "Submission outcome is uncertain for stage $(<"$RUN_ROOT/SUBMISSION_IN_PROGRESS"); inspect the scheduler and jobs.tsv before retrying" >&2
  exit 2
fi
ALREADY_SUBMITTED=0
if [[ -e "$RUN_ROOT/PIPELINE_SUBMITTED" ]]; then
  [[ -n "${JOB_IDS[final_summary]-}" ]] || {
    echo "PIPELINE_SUBMITTED exists but final_summary is missing" >&2; exit 2;
  }
  ALREADY_SUBMITTED=1
fi
EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT,DEEPSC_SHARED_ROOT=$DEEPSC_SHARED_ROOT"

for MODEL in C D; do
  dependency=none
  if [[ "$MODEL" == C ]]; then model_seed=42; else model_seed=43; fi
  for CHUNK in {1..8}; do
    submit_stage "train_${MODEL}_${CHUNK}" "$GPU_PARTITION" "$dependency" \
      --gres="$GPU_RESOURCE" --cpus-per-task=4 --mem=16G --job-name="ds_${MODEL}_${CHUNK}" \
      --output="$RUN_ROOT/logs/train-${MODEL}-${CHUNK}-%j.out" \
      --error="$RUN_ROOT/logs/train-${MODEL}-${CHUNK}-%j.err" \
      --export="$EXPORTS,MODEL_ID=$MODEL,MODEL_SEED=$model_seed,CHUNK_INDEX=$CHUNK" \
      "$PROJECT_ROOT/server/train_chunk.slurm" || exit 1
    dependency="$LAST_JOB_ID"
  done
  if [[ "$MODEL" == C ]]; then C_FINAL="$dependency"; else D_FINAL="$dependency"; fi
done

submit_stage integrity "$GPU_PARTITION" "$C_FINAL:$D_FINAL" --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G \
  --job-name=ds_integrity --output="$RUN_ROOT/logs/integrity-%j.out" \
  --error="$RUN_ROOT/logs/integrity-%j.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/integrity.slurm" || exit 1
INTEGRITY="$LAST_JOB_ID"

submit_stage collect_development "$GPU_PARTITION" "$INTEGRITY" --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G \
  --array=0-59%4 --job-name=ds_collect \
  --output="$RUN_ROOT/logs/collect-%A_%a.out" \
  --error="$RUN_ROOT/logs/collect-%A_%a.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/collect_development.slurm" || exit 1
COLLECT="$LAST_JOB_ID"

submit_stage classical "$CPU_PARTITION" "$COLLECT" --cpus-per-task=4 --mem=32G --job-name=ds_classical \
  --output="$RUN_ROOT/logs/classical-%j.out" \
  --error="$RUN_ROOT/logs/classical-%j.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/extract_and_classical.slurm" || exit 1
CLASSICAL="$LAST_JOB_ID"

submit_stage neural_folds "$GPU_PARTITION" "$COLLECT" --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G \
  --array=0-64%4 --job-name=ds_neural \
  --output="$RUN_ROOT/logs/neural-%A_%a.out" \
  --error="$RUN_ROOT/logs/neural-%A_%a.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/neural_fold.slurm" || exit 1
NEURAL_FOLDS="$LAST_JOB_ID"

submit_stage neural_freeze "$GPU_PARTITION" "$NEURAL_FOLDS" --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G \
  --job-name=ds_neural_freeze \
  --output="$RUN_ROOT/logs/neural-freeze-%j.out" \
  --error="$RUN_ROOT/logs/neural-freeze-%j.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/freeze_neural.slurm" || exit 1
NEURAL_FREEZE="$LAST_JOB_ID"

submit_stage collect_arxiv_holdout "$GPU_PARTITION" "$CLASSICAL:$NEURAL_FREEZE" \
  --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G --array=0-4%2 --job-name=ds_arxiv_collect \
  --output="$RUN_ROOT/logs/arxiv-collect-%A_%a.out" \
  --error="$RUN_ROOT/logs/arxiv-collect-%A_%a.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/collect_holdout.slurm" || exit 1
HOLDOUT="$LAST_JOB_ID"

submit_stage evaluate_arxiv_holdout "$GPU_PARTITION" "$HOLDOUT" --gres="$GPU_RESOURCE" --cpus-per-task=2 --mem=16G \
  --job-name=ds_arxiv_eval \
  --output="$RUN_ROOT/logs/arxiv-eval-%j.out" \
  --error="$RUN_ROOT/logs/arxiv-eval-%j.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/evaluate_holdout.slurm" || exit 1
HOLDOUT_EVAL="$LAST_JOB_ID"

submit_stage final_summary "$CPU_PARTITION" "$HOLDOUT_EVAL" --cpus-per-task=1 --mem=1G --job-name=ds_final \
  --output="$RUN_ROOT/logs/final-%j.out" \
  --error="$RUN_ROOT/logs/final-%j.err" \
  --export="$EXPORTS" "$PROJECT_ROOT/server/final_summary.slurm" || exit 1
if [[ "$ALREADY_SUBMITTED" -eq 1 ]]; then
  printf 'Pipeline already submitted; final summary job %s\n' "$LAST_JOB_ID"
else
  touch "$RUN_ROOT/PIPELINE_SUBMITTED"
  printf 'Submitted pipeline. Final summary job: %s\n' "$LAST_JOB_ID"
fi

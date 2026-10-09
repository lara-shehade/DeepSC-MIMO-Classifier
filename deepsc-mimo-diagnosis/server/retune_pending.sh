#!/bin/bash
# Reduce memory requests on an already-submitted, still-pending run.
set -euo pipefail

if [[ $# -ne 2 || "$2" != --apply ]]; then
  echo "Usage: $0 /absolute/path/to/run --apply" >&2
  exit 2
fi
RUN_ROOT="$(cd "$1" && pwd)"
[[ -f "$RUN_ROOT/preflight/PREFLIGHT_OK" && -f "$RUN_ROOT/PIPELINE_SUBMITTED" ]] || {
  echo "Run must have passed preflight and completed DAG submission" >&2; exit 2;
}
LEDGER="$RUN_ROOT/jobs.tsv"
[[ -f "$LEDGER" ]] || { echo "Missing jobs.tsv" >&2; exit 2; }
IFS= read -r header < "$LEDGER"
[[ "$header" == $'stage\tjob_id\tdependency' ]] || {
  echo "Invalid jobs.tsv header" >&2; exit 2;
}

declare -A SEEN_STAGE=() SEEN_JOB=() TARGET=()
low_ids=""
classical_id=""
final_id=""
count=0
while IFS=$'\t' read -r stage job_id dependency extra; do
  [[ "$job_id" =~ ^[1-9][0-9]*$ && -z "${extra:-}" ]] || {
    echo "Invalid ledger row for $stage" >&2; exit 2;
  }
  [[ -z "${SEEN_STAGE[$stage]-}" && -z "${SEEN_JOB[$job_id]-}" ]] || {
    echo "Duplicate stage or job ID in ledger" >&2; exit 2;
  }
  SEEN_STAGE["$stage"]=1
  SEEN_JOB["$job_id"]=1
  count=$((count + 1))
  case "$stage" in
    preflight)
      [[ "$dependency" == none ]] || { echo "Invalid preflight dependency" >&2; exit 2; }
      continue ;;
    train_C_[1-8]|train_D_[1-8])
      expected_name="ds_${stage#train_}"; memory=16384 ;;
    integrity)
      expected_name=ds_integrity; memory=16384 ;;
    collect_development)
      expected_name=ds_collect; memory=16384 ;;
    classical)
      expected_name=ds_classical; memory=32768 ;;
    neural_folds)
      expected_name=ds_neural; memory=16384 ;;
    neural_freeze)
      expected_name=ds_neural_freeze; memory=16384 ;;
    collect_arxiv_holdout)
      expected_name=ds_arxiv_collect; memory=16384 ;;
    evaluate_arxiv_holdout)
      expected_name=ds_arxiv_eval; memory=16384 ;;
    final_summary)
      expected_name=ds_final; memory=1024 ;;
    *) echo "Unexpected stage in ledger: $stage" >&2; exit 2 ;;
  esac
  status="$(squeue -h -j "$job_id" -o '%T|%j')" || exit 1
  [[ -n "$status" ]] || { echo "$stage ($job_id) is absent from the queue" >&2; exit 2; }
  while IFS='|' read -r state job_name; do
    [[ "$state" == PENDING && "$job_name" == "$expected_name" ]] || {
      echo "$stage ($job_id) is not the expected pending job: $state $job_name" >&2
      exit 2
    }
  done <<< "$status"
  TARGET["$job_id"]="$memory"
  if [[ "$memory" -eq 16384 ]]; then
    low_ids="${low_ids:+$low_ids,}$job_id"
  elif [[ "$memory" -eq 32768 ]]; then
    classical_id="$job_id"
  else
    final_id="$job_id"
  fi
done < <(tail -n +2 "$LEDGER")

[[ "$count" -eq 25 && -n "$low_ids" && -n "$classical_id" && -n "$final_id" ]] || {
  echo "Ledger is incomplete; no resources were changed" >&2; exit 2;
}

# Update the queued jobs themselves. Job IDs and dependencies stay intact.
scontrol update JobId="$low_ids" MinMemoryNode=16384
scontrol update JobId="$classical_id" MinMemoryNode=32768
scontrol update JobId="$final_id" MinMemoryNode=1024

memory_to_mib() {
  local value="$1" number unit
  [[ "$value" =~ ^([0-9]+)([KkMmGgTt]?)$ ]] || return 1
  number="${BASH_REMATCH[1]}"
  unit="${BASH_REMATCH[2]}"
  case "$unit" in
    ""|M|m) printf '%s' "$number" ;;
    G|g) printf '%s' "$((number * 1024))" ;;
    T|t) printf '%s' "$((number * 1024 * 1024))" ;;
    K|k) [[ "$((number % 1024))" -eq 0 ]] || return 1
         printf '%s' "$((number / 1024))" ;;
  esac
}

while IFS=$'\t' read -r stage job_id dependency extra; do
  [[ "$stage" == preflight ]] && continue
  details="$(scontrol show job -o "$job_id")" || exit 1
  [[ "$details" =~ (^|[[:space:]])MinMemoryNode=([^[:space:]]+) ]] || {
    echo "Cannot verify memory request for $stage ($job_id)" >&2; exit 1;
  }
  displayed_memory="${BASH_REMATCH[2]}"
  actual_mib="$(memory_to_mib "$displayed_memory")" || {
    echo "Unknown memory format for $stage ($job_id): $displayed_memory" >&2; exit 1;
  }
  [[ "$actual_mib" -eq "${TARGET[$job_id]}" ]] || {
    echo "Memory mismatch for $stage ($job_id): expected ${TARGET[$job_id]} MiB, Slurm shows $displayed_memory" >&2; exit 1;
  }
done < <(tail -n +2 "$LEDGER")

MANIFEST="$RUN_ROOT/resource_overrides.tsv"
if [[ ! -e "$MANIFEST" ]]; then
  printf 'utc_time\tstage\tjob_id\tnew_memory_mib\n' > "$MANIFEST"
fi
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while IFS=$'\t' read -r stage job_id dependency extra; do
  [[ "$stage" == preflight ]] && continue
  printf '%s\t%s\t%s\t%s\n' "$timestamp" "$stage" "$job_id" "${TARGET[$job_id]}" >> "$MANIFEST"
done < <(tail -n +2 "$LEDGER")
echo "Updated and verified 24 pending jobs; no jobs were canceled or resubmitted"

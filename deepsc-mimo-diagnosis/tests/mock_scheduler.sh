#!/bin/bash
set -euo pipefail
tool="${0##*/}"
case "$tool" in
  squeue)
    job_id=""
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == -j ]]; then job_id="$2"; shift 2; else shift; fi
    done
    stage="$(awk -F '\t' -v id="$job_id" '$2==id {print $1}' "$MOCK_RUN_ROOT/jobs.tsv")"
    case "$stage" in
      train_*) name="ds_${stage#train_}" ;;
      integrity) name=ds_integrity ;;
      collect_development) name=ds_collect ;;
      classical) name=ds_classical ;;
      neural_folds) name=ds_neural ;;
      neural_freeze) name=ds_neural_freeze ;;
      collect_arxiv_holdout) name=ds_arxiv_collect ;;
      evaluate_arxiv_holdout) name=ds_arxiv_eval ;;
      final_summary) name=ds_final ;;
      *) exit 1 ;;
    esac
    if [[ "$stage" == "${MOCK_RUNNING_STAGE:-}" ]]; then
      printf 'RUNNING|%s\n' "$name"
    else
      printf 'PENDING|%s\n' "$name"
    fi ;;
  scontrol)
    mode="$1"; shift
    if [[ "$mode" == update ]]; then
      ids="${1#JobId=}"; memory="${2#MinMemoryNode=}"
      if [[ "$memory" == "${MOCK_FAIL_UPDATE_ON:-}" ]]; then exit 42; fi
      IFS=, read -ra id_list <<< "$ids"
      for id in "${id_list[@]}"; do
        printf '%s %s\n' "$id" "$memory" >> "$MOCK_MEMORY"
      done
    elif [[ "$mode" == show && "$1" == job && "$2" == -o ]]; then
      id="$3"
      memory="$(awk -v id="$id" '$1==id {value=$2} END {print value}' "$MOCK_MEMORY")"
      if [[ "${MOCK_MEMORY_SUFFIX:-}" == G ]]; then
        [[ "$((memory % 1024))" -eq 0 ]] || exit 1
        memory="$((memory / 1024))G"
      fi
      printf 'JobId=%s JobState=PENDING MinMemoryNode=%s\n' "$id" "$memory"
    else
      exit 1
    fi ;;
  *) exit 1 ;;
esac

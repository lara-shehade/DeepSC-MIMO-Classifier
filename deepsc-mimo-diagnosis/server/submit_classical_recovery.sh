#!/bin/bash
# Recover a timed-out classical stage without repeating feature extraction.
set -euo pipefail

[[ $# -eq 2 && ( "$2" == --check || "$2" == --apply ) ]] || {
  echo "Usage: $0 /absolute/run/root --check|--apply" >&2; exit 2;
}
MODE="$2"
RUN_ROOT="$(cd "$1" && pwd)"
PROJECT_ROOT="$RUN_ROOT/code"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$RUN_ROOT/jobs.tsv" && -f "$PROJECT_ROOT/analysis/classical_diagnostics.py" ]] || {
  echo "Not a DeepSC-MIMO run root: $RUN_ROOT" >&2; exit 2;
}
exec 9> "$RUN_ROOT/.classical_recovery.lock"
flock -n 9 || { echo "Another classical recovery submission is active" >&2; exit 2; }
LEDGER="$RUN_ROOT/classical_recovery_jobs.tsv"
GUARD="$RUN_ROOT/CLASSICAL_RECOVERY_SUBMISSION_IN_PROGRESS"
if [[ -f "$LEDGER" && -f "$RUN_ROOT/CLASSICAL_RECOVERY_SUBMITTED" ]]; then
  echo "Recovery was already submitted; see $LEDGER"
  exit 0
fi
[[ ! -e "$LEDGER" && ! -e "$GUARD" ]] || {
  echo "Previous recovery submission may be incomplete; inspect $LEDGER and $GUARD" >&2
  exit 2
}
[[ -s "$RUN_ROOT/analysis/development_features.csv" ]] || {
  echo "Missing completed development feature CSV" >&2; exit 2;
}
[[ -f "$RUN_ROOT/analysis/frozen_neural.FROZEN" ]] || {
  echo "Neural model is not frozen" >&2; exit 2;
}
[[ ! -e "$RUN_ROOT/analysis/classical/FROZEN" ]] || {
  echo "Classical model is already frozen" >&2; exit 2;
}
CLASSICAL_JOB="$(awk -F '\t' '$1=="classical"{print $2}' "$RUN_ROOT/jobs.tsv")"
HOLDOUT_JOB="$(awk -F '\t' '$1=="collect_arxiv_holdout"{print $2}' "$RUN_ROOT/jobs.tsv")"
[[ "$CLASSICAL_JOB" =~ ^[1-9][0-9]*$ && "$HOLDOUT_JOB" =~ ^[1-9][0-9]*$ ]] || {
  echo "Missing classical or holdout job ID in jobs.tsv" >&2; exit 2;
}
CLASSICAL_STATE="$(sacct -X -n -P -j "$CLASSICAL_JOB" -o State)"
[[ "$CLASSICAL_STATE" == TIMEOUT* ]] || {
  echo "Original classical job is not TIMEOUT: $CLASSICAL_STATE" >&2; exit 2;
}
HOLDOUT_STATE="$(squeue -h -j "$HOLDOUT_JOB" -o %T)"
[[ "$HOLDOUT_STATE" == PENDING ]] || {
  echo "Existing arXiv array is not pending: $HOLDOUT_STATE" >&2; exit 2;
}

# Install only the two new recovery files into this run's source snapshot.
for NAME in classical_recovery.py classical_recovery.slurm; do
  SOURCE="$SOURCE_ROOT/server/$NAME"
  TARGET="$PROJECT_ROOT/server/$NAME"
  [[ -f "$SOURCE" ]] || { echo "Missing $SOURCE" >&2; exit 2; }
  if [[ -e "$TARGET" ]]; then
    cmp -s "$SOURCE" "$TARGET" || {
      echo "Run already has a different $TARGET; refusing overwrite" >&2; exit 2;
    }
  else
    cp "$SOURCE" "$TARGET"
  fi
done
(
  cd "$RUN_ROOT"
  sha256sum code/server/classical_recovery.py code/server/classical_recovery.slurm \
    > classical_recovery_sources.sha256
)

# A tiny synthetic-data equivalence test must pass before any sbatch call.
module purge
module load python/ai-4
cd "$PROJECT_ROOT"
if PYTHONPATH="$PROJECT_ROOT" python "$SOURCE_ROOT/tests/classical_smoke.py" \
  > "$RUN_ROOT/logs/classical-recovery-smoke.log" 2>&1; then
  echo "Classical split equivalence test passed"
else
  tail -n 15 "$RUN_ROOT/logs/classical-recovery-smoke.log" >&2
  echo "No recovery jobs submitted" >&2
  exit 1
fi
if [[ "$MODE" == --check ]]; then
  echo "Recovery preflight passed; no jobs submitted"
  exit 0
fi

printf 'starting\n' > "$GUARD"
PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"
EXPORT_BASE="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ROOT=$RUN_ROOT"
if ARRAY_OUTPUT="$(sbatch --parsable --partition="$PARTITION" \
  --array=0-24%2 --cpus-per-task=2 --mem=16G \
  --job-name=ds_classical_fold \
  --output="$RUN_ROOT/logs/classical-fold-%A_%a.out" \
  --error="$RUN_ROOT/logs/classical-fold-%A_%a.err" \
  --export="$EXPORT_BASE,CLASSICAL_RECOVERY_MODE=fold" \
  "$PROJECT_ROOT/server/classical_recovery.slurm")"; then :; else
  echo "Classical fold sbatch failed; inspect $GUARD before retrying" >&2
  exit 1
fi
[[ "$ARRAY_OUTPUT" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
  echo "Invalid fold sbatch response: $ARRAY_OUTPUT" >&2; exit 1;
}
ARRAY_JOB="${BASH_REMATCH[1]}"
printf 'stage\tjob_id\tdependency\nfolds\t%s\tnone\n' "$ARRAY_JOB" > "$LEDGER"

if FINAL_OUTPUT="$(sbatch --parsable --partition="$PARTITION" \
  --dependency="afterok:$ARRAY_JOB" --cpus-per-task=2 --mem=16G \
  --job-name=ds_classical_final \
  --output="$RUN_ROOT/logs/classical-final-%j.out" \
  --error="$RUN_ROOT/logs/classical-final-%j.err" \
  --export="$EXPORT_BASE,CLASSICAL_RECOVERY_MODE=finalize" \
  "$PROJECT_ROOT/server/classical_recovery.slurm")"; then :; else
  echo "Finalization sbatch failed; fold array is $ARRAY_JOB; inspect $LEDGER" >&2
  exit 1
fi
[[ "$FINAL_OUTPUT" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || {
  echo "Invalid finalization sbatch response: $FINAL_OUTPUT" >&2; exit 1;
}
FINAL_JOB="${BASH_REMATCH[1]}"
printf 'finalize\t%s\tafterok:%s\n' "$FINAL_JOB" "$ARRAY_JOB" >> "$LEDGER"

# Neural freeze has completed, so only the replacement classical finalizer
# remains as a prerequisite for the original, still-pending arXiv array.
if ! scontrol update JobId="$HOLDOUT_JOB" Dependency="afterok:$FINAL_JOB"; then
  echo "Could not reconnect arXiv job $HOLDOUT_JOB; fold/final IDs are in $LEDGER" >&2
  exit 1
fi
HOLDOUT_RECORD="$(scontrol show job -o "$HOLDOUT_JOB")"
if [[ "$HOLDOUT_RECORD" != *"Dependency=afterok:$FINAL_JOB"* ]]; then
  echo "ArXiv dependency update could not be verified; inspect job $HOLDOUT_JOB" >&2
  exit 1
fi
printf 'holdout\t%s\tafterok:%s\n' "$HOLDOUT_JOB" "$FINAL_JOB" >> "$LEDGER"
mv "$GUARD" "$RUN_ROOT/CLASSICAL_RECOVERY_SUBMITTED"
printf 'Submitted classical folds %s and finalizer %s; reconnected arXiv %s\n' \
  "$ARRAY_JOB" "$FINAL_JOB" "$HOLDOUT_JOB"

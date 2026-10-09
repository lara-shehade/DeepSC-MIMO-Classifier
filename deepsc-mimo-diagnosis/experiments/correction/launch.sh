#!/bin/bash
# correction_v1: diagnosis-gated restoration experiments. CPU only.
# Reads the existing run (traces, checkpoints, fusion probabilities). Writes ONLY to
#   $RUN_ROOT/correction_v1/
# Usage (from the repository root): RUN_ROOT=... DEEPSC_SHARED_ROOT=... bash experiments/correction/launch.sh
# Chain: D->D targets -> smoke fold (seed 3101, 2 batches) -> 5 full folds -> summary.
# Each stage starts only if the previous one succeeded.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a pipeline run folder (contains traces/, checkpoints/, code/)}"
SHARED_ROOT="${DEEPSC_SHARED_ROOT:?set DEEPSC_SHARED_ROOT to the shared data folder}"
PROBS="${PROBS_DIR:-$RUN_ROOT/paper_extraction_v1/probs}"
PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"
BUNDLE_DIR="${BUNDLE_DIR:-$RUN_ROOT}"
MAIL_OPT=""; [ -n "${MAIL_USER:-}" ] && MAIL_OPT="--mail-user=$MAIL_USER --mail-type=FAIL"
OUT="$RUN_ROOT/correction_v1"
HERE="$(cd "$(dirname "$0")" && pwd)"

for p in "$RUN_ROOT/traces/development" "$RUN_ROOT/code/collect_profiles.py" \
         "$RUN_ROOT/checkpoints/C/last.pth" "$RUN_ROOT/checkpoints/D/last.pth" \
         "$PROBS/fusion_probs_seed3101.csv" "$SHARED_ROOT"; do
  test -e "$p" || { echo "missing $p"; exit 1; }
done
if [ -e "$OUT" ]; then echo "$OUT already exists; rename it first (nothing was submitted)"; exit 1; fi

mkdir -p "$OUT"/{code,jobs,logs,folds}
cp "$HERE"/common.py "$HERE"/restoration_maps.py "$HERE"/make_dd.py "$HERE"/run_fold.py "$HERE"/summarize.py "$OUT/code/"
(cd "$OUT/code" && sha256sum *.py > ../code.sha256)

HEADER="#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
(cd $OUT/code && sha256sum -c ../code.sha256 >/dev/null)
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
cd $OUT/code
echo \"RUN_ROOT=$RUN_ROOT  OUT=$OUT  host=\$(hostname)\""
COMMON="--run-root $RUN_ROOT --shared-root $SHARED_ROOT --output $OUT"

cat > "$OUT/jobs/dd.sh" <<EOF
$HEADER
python make_dd.py $COMMON --batches 10
EOF
cat > "$OUT/jobs/smoke.sh" <<EOF
$HEADER
python run_fold.py $COMMON --probs-dir $PROBS --held-out-seed 3101 --smoke --threads \$SLURM_CPUS_PER_TASK
python summarize.py --output $OUT --smoke
EOF
cat > "$OUT/jobs/fold.sh" <<EOF
$HEADER
SEEDS=(3101 3102 3103 3104 3105)
python run_fold.py $COMMON --probs-dir $PROBS --held-out-seed \${SEEDS[\$SLURM_ARRAY_TASK_ID]} --threads \$SLURM_CPUS_PER_TASK
EOF
cat > "$OUT/jobs/summary.sh" <<EOF
$HEADER
python summarize.py --output $OUT
cd $RUN_ROOT && tar -czf $BUNDLE_DIR/correction_v1_results.tgz --exclude=correction_v1/code --exclude='correction_v1/logs/summary-*' correction_v1
echo "bundle: $BUNDLE_DIR/correction_v1_results.tgz"
EOF

SB="sbatch --parsable --partition=$PARTITION $MAIL_OPT"
DD=$($SB --cpus-per-task=2 --mem=4G --time=01:00:00 --job-name=corr_dd \
  --output="$OUT/logs/dd-%j.out" --error="$OUT/logs/dd-%j.err" "$OUT/jobs/dd.sh")
SMOKE=$($SB --cpus-per-task=4 --mem=8G --time=01:00:00 --dependency=afterok:$DD --job-name=corr_smoke \
  --output="$OUT/logs/smoke-%j.out" --error="$OUT/logs/smoke-%j.err" "$OUT/jobs/smoke.sh")
FOLDS=$($SB --cpus-per-task=4 --mem=8G --time=05:00:00 --array=0-4 --dependency=afterok:$SMOKE \
  --job-name=corr_fold --output="$OUT/logs/fold-%A_%a.out" --error="$OUT/logs/fold-%A_%a.err" "$OUT/jobs/fold.sh")
SUMMARY=$($SB --cpus-per-task=1 --mem=2G --time=00:20:00 --dependency=afterok:$FOLDS --job-name=corr_summary \
  --output="$OUT/logs/summary-%j.out" --error="$OUT/logs/summary-%j.err" "$OUT/jobs/summary.sh")
printf "stage\tjob\ndd\t%s\nsmoke\t%s\nfolds\t%s\nsummary\t%s\n" "$DD" "$SMOKE" "$FOLDS" "$SUMMARY" | tee "$OUT/jobs.tsv"
echo "Smoke result (after ~1 h): $OUT/summary_smoke.md   Final bundle: $BUNDLE_DIR/correction_v1_results.tgz"

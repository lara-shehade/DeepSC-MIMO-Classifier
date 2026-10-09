#!/bin/bash
# Paper extraction (CPU only): per-sentence quality metrics, per-sentence fusion
# probabilities for the five held-out seeds, and their analysis.
# This is the submitter that produced jobs/*.sh in the paper's run, with the
# cluster paths and e-mail replaced by environment variables.
#
# From the repository root:
#   export RUN_ROOT=/path/to/runs/<run_id>           # finished pipeline run
#   export DEEPSC_SHARED_ROOT=/path/to/adv_datasets
#   bash experiments/paper_extraction/launch.sh
# Optional: DEEPSC_CPU_PARTITION (default gpu), MAIL_USER.
# Writes only to $RUN_ROOT/paper_extraction_v1/.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a finished pipeline run folder}"
SHARED_ROOT="${DEEPSC_SHARED_ROOT:?set DEEPSC_SHARED_ROOT to the adv_datasets folder}"
PARTITION="${DEEPSC_CPU_PARTITION:-gpu}"
MAIL_OPT=()
[ -n "${MAIL_USER:-}" ] && MAIL_OPT=(--mail-user="$MAIL_USER" --mail-type=FAIL)

OUT="$RUN_ROOT/paper_extraction_v1"
HERE="$(cd "$(dirname "$0")" && pwd)"

test -d "$RUN_ROOT/traces/development" || { echo "missing $RUN_ROOT/traces/development"; exit 1; }
test -d "$RUN_ROOT/code/analysis"      || { echo "missing $RUN_ROOT/code/analysis"; exit 1; }
test -d "$RUN_ROOT/analysis/neural_folds" || { echo "missing $RUN_ROOT/analysis/neural_folds"; exit 1; }
if [ -e "$OUT" ]; then echo "$OUT already exists; remove or rename it first"; exit 1; fi

mkdir -p "$OUT"/{code,jobs,metrics,probs,analysis,logs}
cp "$HERE"/paper_metrics.py "$HERE"/fusion_probs.py "$OUT/code/"
sha256sum "$OUT"/code/*.py > "$OUT/code.sha256"

HEADER="#!/bin/bash
set -euo pipefail
module purge
module load python/ai-4
export PYTHONPATH=$RUN_ROOT/code
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
cd $RUN_ROOT/code"

cat > "$OUT/jobs/metrics.sh" <<EOF
$HEADER
python $OUT/code/paper_metrics.py --run-root $RUN_ROOT --shared-root $SHARED_ROOT --output-dir $OUT/metrics
EOF

cat > "$OUT/jobs/fold.sh" <<EOF
$HEADER
SEEDS=(3101 3102 3103 3104 3105)
S=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
python $OUT/code/fusion_probs.py --mode fold --trace-root $RUN_ROOT/traces/development \\
  --held-out-seed \$S --device cpu --output $OUT/probs/fusion_probs_seed\$S.csv
EOF

cat > "$OUT/jobs/analyze.sh" <<EOF
$HEADER
python $OUT/code/fusion_probs.py --mode analyze --probs-dir $OUT/probs \\
  --original-folds $RUN_ROOT/analysis/neural_folds \\
  --batch-channel $OUT/metrics/batch_channel.csv --output-dir $OUT/analysis
EOF

METRICS=$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=1 --mem=3G --time=00:30:00 \
  --job-name=ds_paper_metrics ${MAIL_OPT[@]+"${MAIL_OPT[@]}"} \
  --output="$OUT/logs/metrics-%j.out" --error="$OUT/logs/metrics-%j.err" "$OUT/jobs/metrics.sh")

FOLDS=$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=2 --mem=8G --time=01:30:00 \
  --array=0-4 --job-name=ds_paper_probs ${MAIL_OPT[@]+"${MAIL_OPT[@]}"} \
  --output="$OUT/logs/probs-%A_%a.out" --error="$OUT/logs/probs-%A_%a.err" "$OUT/jobs/fold.sh")

ANALYZE=$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=1 --mem=1G --time=00:15:00 \
  --dependency="afterok:$METRICS:$FOLDS" --job-name=ds_paper_analyze ${MAIL_OPT[@]+"${MAIL_OPT[@]}"} \
  --output="$OUT/logs/analyze-%j.out" --error="$OUT/logs/analyze-%j.err" "$OUT/jobs/analyze.sh")

printf "stage\tjob_id\nmetrics\t%s\nfolds\t%s\nanalyze\t%s\n" "$METRICS" "$FOLDS" "$ANALYZE" | tee "$OUT/jobs.tsv"
echo "Outputs: $OUT/metrics, $OUT/probs, $OUT/analysis"

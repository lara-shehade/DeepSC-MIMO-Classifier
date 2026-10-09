#!/bin/bash
# Re-run the diagnosis part of the pipeline on the released data and compare with results/.
# No DeepSC training or trace collection is repeated. CPU only.
#
# From the repository root:
#   export FEATURES=/path/to/development_features.csv   # released feature table
#   export TRACES=/path/to/traces/development           # released receiver traces (R0-R3)
#   bash verify/run_verification.sh quick          # one fold, fusion elastic net      (needs FEATURES)
#   bash verify/run_verification.sh classical      # all 75 classical folds            (needs FEATURES)
#   bash verify/run_verification.sh features       # re-extract the 972 features       (needs TRACES)
#   bash verify/run_verification.sh localization   # first-divergence on 400 batches   (needs TRACES)
#   bash verify/run_verification.sh neural [SEED]  # one fusion fold, default 3101     (needs TRACES)
# Outputs go to $OUT (default ./verify_out).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-verify_out}"
mkdir -p "$OUT"
STEP="${1:?step: quick | classical | features | localization | neural}"

case "$STEP" in
  quick)
    python verify/compare.py classical-quick "${FEATURES:?set FEATURES}" ;;
  classical)
    python analysis/classical_diagnostics.py "${FEATURES:?set FEATURES}" --output-root "$OUT/classical"
    python verify/compare.py classical "$OUT/classical/grouped_results.json" ;;
  features)
    python analysis/extract_sentence_features.py "${TRACES:?set TRACES}" --output "$OUT/development_features.csv"
    if [ -n "${FEATURES:-}" ]; then
      python verify/compare.py features "$OUT/development_features.csv" "$FEATURES"
    fi ;;
  localization)
    python analysis/first_divergence.py "${TRACES:?set TRACES}" --output "$OUT/first_divergence.json"
    python verify/compare.py localization "$OUT/first_divergence.json" ;;
  neural)
    SEED="${2:-3101}"
    python analysis/neural_diagnostics.py --mode fold --trace-root "${TRACES:?set TRACES}" \
      --architecture fusion --stage fusion --held-out-seed "$SEED" --device cpu \
      --output "$OUT/fusion_fusion_seed$SEED.json"
    python verify/compare.py neural "$OUT/fusion_fusion_seed$SEED.json" ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac

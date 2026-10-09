# Verifying the diagnosis results without retraining

These scripts re-run everything **after** trace collection on the released data, then compare the output with the stored files in `results/`:
- feature extraction;
- localization;
- the classical diagnostics;
- a neural fold.

DeepSC is not trained again and no traces are collected again. Everything runs on CPU.

## Data

The feature table and traces are not part of the repository or the dataset handoff. Request them from the authors, or produce them by running the pipeline. Then point the scripts at them:

```
development_features.csv     # 44,800 sentences x 972 features (+ metadata)
traces/development/          # receiver traces R0-R3 for every condition, seed and batch
```

```bash
export FEATURES=/path/to/development_features.csv
export TRACES=/path/to/traces/development
```

## Steps (from the repository root)

| Command | Needs | Checks against |
|---|---|---|
| `bash verify/run_verification.sh quick` | FEATURES | one fold (fusion, elastic net, held-out seed 3101) in `results/classical/` |
| `bash verify/run_verification.sh classical` | FEATURES | all 75 classical folds (5 inputs × 3 pipelines × 5 seeds) |
| `bash verify/run_verification.sh features` | TRACES | re-extracts the 972 features and compares them with FEATURES, sentence by sentence |
| `bash verify/run_verification.sh localization` | TRACES | the entry stage of all 400 paired batches |
| `bash verify/run_verification.sh neural 3101` | TRACES | one fusion fold in `results/neural/folds/` |

Outputs go to `./verify_out` (or `$OUT`). Each comparison ends with **PASS** or **CHECK**.

## What to expect

- **Features and localization:** these are deterministic, so they should match exactly.
- **Classical folds:** they should match within 0.01 macro-F1.
  - The `saga` and `liblinear` solvers shuffle internally.
  - Very different scikit-learn versions can also shift results slightly.
- **Neural fold:** it should match within about 0.02, because the original folds were trained on a V100 GPU and this check trains on CPU.
- **Memory:** the classical step loads the whole feature table, so plan for about 16 GB of RAM. The original cluster job requested 32 GB.

`requirements.txt` gives minimum versions. For the closest match, use the versions from the cluster run: Python 3, PyTorch, NumPy and scikit-learn as loaded by `module load python/ai-4`.

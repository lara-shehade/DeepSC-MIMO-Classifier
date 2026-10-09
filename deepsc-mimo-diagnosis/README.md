# Root-Cause Diagnosis for MIMO Semantic Communications

Code for the letter *"Root-Cause Diagnosis for MIMO Semantic Communications"*, by Lara Chehade (AUB VIP, supervised by Dr. Hadi Sarieddeen).

The system is a DeepSC transceiver over a 2×2 MIMO Rayleigh channel with an LMMSE receiver. The diagnostic sees only four receiver checkpoints:
- **R0**, the received signal;
- **R1**, the equalized symbols;
- **R2**, the channel-decoder output;
- **R3**, the decoder's hidden states.

From these it identifies which of seven conditions is present: nominal, low SNR, imperfect CSI, spatial correlation, literal corruption, model mismatch, or out-of-domain text.

The repository covers:
- training two DeepSC models (C and D);
- paired collection of the seven conditions;
- first-divergence localization;
- 972 engineered features per sentence;
- classical diagnostics (ANOVA, mRMR and elastic net), with the neural CNN, TCN, conv-Transformer and fusion models in the same evaluation;
- the frozen arXiv holdout test;
- Shapley stage attribution;
- the observation-window study;
- diagnosis-gated representation restoration.

A file-by-file explanation of the code is in [`docs/DeepSC_MIMO_Implementation_Report.pdf`](docs/DeepSC_MIMO_Implementation_Report.pdf).

## Repository layout

| Path | Contents |
|---|---|
| `main.py`, `models/`, `utils.py`, `channels.py` | DeepSC model, training (exact RNG resume), MIMO channel (Rayleigh, imperfect CSI, Kronecker correlation, LMMSE) |
| `profiles.py`, `collect_profiles.py` | The seven conditions and paired trace collection (R0–R3) |
| `analysis/` | Localization, feature extraction, classical and neural diagnostics, holdout evaluation |
| `stage_subset/` | The 15 stage-subset ablation runs |
| `classical_tail/`, `classical_freeze_recovery/` | Recovery steps that completed the classical stage under a 6-hour job limit (see `docs/DEPLOYMENT.md`) |
| `server/` | Slurm job scripts, `launch_pipeline.sh` (whole job graph), and the run-repair scripts used during the paper's run |
| `config/` | Pipeline configuration |
| `tests/` | Unit tests and synthetic smoke tests (no data needed) |
| `verify/` | Re-run the diagnosis on the released data and compare with `results/` |
| `experiments/` | Add-on analyses run on a finished pipeline run (see below) |
| `results/` | The small result files behind the paper's tables |
| `docs/` | Implementation report, deployment and resource notes |
| `provenance/` | Hash lists from the cluster run, a script that checks this repository against them, and what happened during the run |

## Installation

Python 3.10. The paper's run used Python 3.10.13, PyTorch 2.1.2 and CUDA 12.1 on a V100 32 GB, as recorded in the run's preflight report.

```bash
pip install -r requirements.txt
```

## Quick check (CPU, a few minutes, no data needed)

```bash
python -m pytest tests/
python -m tests.integration_smoke     # synthetic train → collect → extract
python -m tests.classical_smoke       # synthetic classical diagnostics
```

## Numbers that can be recomputed without data

```bash
# Shapley values, interaction indices and Kendall's W (Section IV)
python experiments/stage_attribution/stage_attribution_analysis.py \
    results/stage_subset/stage_subset_report.json shapley.json

# How far imperfect CSI can be reversed (Section V)
python experiments/csi_analytic_check/csi_analytic_check.py
python experiments/window_analysis/bayes_corr.py      # genie test, nominal vs correlation
python experiments/window_analysis/bayes_m.py
```

## Data

The prepared datasets and shared vocabulary are included in the private handoff as the adjacent `../adv_datasets/` directory. Checkpoints are not included because the full pipeline trains models C and D from scratch. The pipeline expects this layout under `DEEPSC_SHARED_ROOT`:

```
DEEPSC_SHARED_ROOT/
├── shared_vocab.json          # 90,746 tokens; PAD/START/END/UNK = 0/1/2/3
├── europarl/   train_data.pkl  test_data.pkl   # training and in-domain test
├── ukhansard/  train_data.pkl  test_data.pkl   # out-of-domain condition
└── arxiv/      train_data.pkl  test_data.pkl   # frozen holdout, used once at the end
```

- **Pickles:** tokenized with `preprocess_text.py`, the original DeepSC preprocessing.
- **Vocabulary:** all three corpora are encoded with the same shared vocabulary. Its size is checked at start-up; `--allow-nonshared-vocab` exists for tests only.
- **Access:** in the complete private handoff, set `DEEPSC_SHARED_ROOT` to the included adjacent `adv_datasets` directory.

## Verifying the results without retraining

`verify/` re-runs feature extraction, localization, the classical diagnostics and a neural fold on the released feature table and traces, and compares each output with `results/`. No DeepSC training is repeated. See [`verify/README.md`](verify/README.md).

The feature table and receiver traces needed for this are not in the repository because of their size. They are available from the authors on request.

## Running the full pipeline (Slurm)

The job scripts were written for the AUB HPC cluster:
- they call `module load python/ai-4`;
- they request one V100 per GPU job.

Adjust the module line and the variables below for another cluster.

```bash
export DEEPSC_SHARED_ROOT="$(cd ../adv_datasets && pwd -P)"
export DEEPSC_RUNS_ROOT=/path/to/a/writable/results/directory
# optional: DEEPSC_GPU_PARTITION, DEEPSC_CPU_PARTITION, DEEPSC_GPU_RESOURCE (default gpu:v100d32q:1)

bash server/launch_pipeline.sh preflight            # checks data, code and resources; creates a run folder
bash server/launch_pipeline.sh continue /abs/path/to/runs/<run_id>   # only after a clean preflight
```

The launcher copies the code into `<run>/code` and submits the dependent jobs in this order:
1. train C and D (80 epochs, in 8 resumable chunks);
2. the C/D integrity check;
3. development trace collection;
4. feature extraction, localization and the classical diagnostics;
5. the 65 neural folds;
6. the neural freeze;
7. the arXiv holdout;
8. the final summary.

**Job time limit:** on a cluster with a 6-hour limit, the classical stage (step 4) does not fit in one job. `docs/DEPLOYMENT.md` §3 gives two things: the general recovery, and what was done in the paper's run. Without a time limit, the classical stage is a single command.

Resources are in `docs/RESOURCES.md`.

## Add-on experiments (run on a finished pipeline run)

Each one reads the run folder and writes only inside it. All are CPU-only except the stage-subset ablation, which uses GPU jobs. `experiments/*/jobs/` holds the exact job scripts that ran in the paper's run, with paths replaced by `${RUN_ROOT}` and `${DEEPSC_SHARED_ROOT}`. The launchers regenerate the same scripts.

| Folder | What it does | How to run |
|---|---|---|
| `experiments/paper_extraction/` | Per-sentence BLEU and token accuracy (Table I); fusion probabilities per fold; calibration, observation window, condition number | `RUN_ROOT=... DEEPSC_SHARED_ROOT=... bash experiments/paper_extraction/launch.sh` (metrics, then 5 folds, then analyze) |
| `stage_subset/` | The 15 stage-subset ablation runs (TCN branches) | see `stage_subset/README.md` |
| `experiments/stage_attribution/` | Shapley values and interactions from the stage-subset report | see above |
| `experiments/window_analysis/` | Combination rules, genie Bayes tests, binary nominal-vs-correlation test, cross-batch pooling | `python experiments/window_analysis/<script>.py <probs_dir> <out.json>`, where `probs_dir` = `$RUN_ROOT/paper_extraction_v1/probs`; `bayes_corr.py` and `bayes_m.py` take no arguments |
| `experiments/correction/` | Diagnosis-gated restoration (Section V) | after paper extraction: `RUN_ROOT=... DEEPSC_SHARED_ROOT=... bash experiments/correction/launch.sh` (see its README) |
| `experiments/csi_analytic_check/` | Analytic limit of CSI reversal | see above |

## Results (`results/`)

| File | Contents |
|---|---|
| `classical/grouped_results.json` | All classical folds (5 inputs × 3 pipelines × 5 held-out seeds), with confusion matrices |
| `neural/frozen_neural.selection.json`, `neural/folds/` | All 65 neural folds and the frozen choice (fusion) |
| `localization/first_divergence.json` | The 400 paired batches and their entry stages |
| `stage_subset/stage_subset_report.json` | 15 stage subsets × 5 folds |
| `paper_extraction/` | Quality per condition, per-batch channel statistics, calibration, window and condition-number analyses |
| `correction/` | Restoration results (`summary.md` and the 5 fold files) |
| `arxiv_holdout.json` | arXiv holdout, scored once after both diagnostics were frozen. It is byte-identical to the run's file (hash recorded in the run's final summary). |

Machine-specific paths in these files are replaced by `<RUN_ROOT>`.

## Reproducibility

- **Seeds:**
  - models C and D use 42 and 43;
  - the physical seeds are 3101–3105;
  - the channel, noise and CSI-error draws use separate random streams, so all conditions share the nominal channel and noise draws.
- **Evaluation:** leave one physical seed out, with balanced classes (35,840 training and 8,960 test sentences per fold). Imputation, scaling and feature selection are fitted on the training folds only.
- **A fresh run:** it uses the same seeds and deterministic cuDNN settings, but GPU arithmetic can differ slightly across hardware and library versions. Expect close numbers, not identical ones. The stored `results/` are the reference.

## Provenance

Every Python file here is identical to the copy that produced the paper's results. The check against the run's own SHA-256 lists is one command:

```bash
python provenance/check_hashes.py
```

`provenance/README.md` records:
- the run-time fixes (the RNG-resume fix, the corrected launcher, the lowered memory requests);
- how the classical stage was completed;
- which analyses were run offline: stage attribution, window analysis, the CSI check, and the pooled correction summary.

Repository-only additions, not part of the run:
- this README and `.gitignore`;
- `verify/`;
- `provenance/check_hashes.py`;
- `experiments/correction/launch.sh`;
- `experiments/paper_extraction/launch.sh` (rebuilt from the original submit script; it regenerates the same job scripts).

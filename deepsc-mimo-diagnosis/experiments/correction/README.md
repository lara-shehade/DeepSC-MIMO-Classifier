# correction_v1: does diagnosis-gated restoration work?

## Run (one command, CPU only, writes only inside the run folder)

From the repository root, point it at a finished pipeline run and run:

```bash
export RUN_ROOT=/path/to/runs/<run_id>        # contains traces/, checkpoints/, code/
export DEEPSC_SHARED_ROOT=/path/to/adv_datasets
bash experiments/correction/launch.sh
```

Run `experiments/paper_extraction/launch.sh` first: the routing stage needs its fusion probabilities.

Optional: `PROBS_DIR` (default `$RUN_ROOT/paper_extraction_v1/probs`), `BUNDLE_DIR` (default `$RUN_ROOT`), `MAIL_USER`, `DEEPSC_CPU_PARTITION`.

The command submits four stages. Each starts only if the previous one succeeded:

| Stage | Resources | What it does |
|---|---|---|
| `corr_dd` | 2 CPU, 4 GB, ≤1 h | Builds the D→D target for mismatch: the same sentences, channel draw and noise, sent and received by D. It verifies that H and the noise match the stored normal artifacts exactly. |
| `corr_smoke` | 4 CPU, 8 GB, ≤1 h | One quick fold (seed 3101, 2 batches, 30 training steps) → `correction_v1/summary_smoke.md` |
| `corr_fold` ×5 | 4 CPU, 8 GB, ≤5 h each | The full leave-one-physical-seed-out folds |
| `corr_summary` | 1 CPU, 2 GB | Tables → `correction_v1/summary.md`, plus the bundle `$BUNDLE_DIR/correction_v1_results.tgz` |

Inputs (read only):
- `RUN/traces/development` (the paired collection)
- `RUN/checkpoints/{C,D}/last.pth`
- `RUN/paper_extraction_v1/probs` (the diagnostic's held-out probabilities)
- the frozen code in `RUN/code`

## Files

- `common.py`, `restoration_maps.py`, `make_dd.py`, `run_fold.py` and `summarize.py` are the exact files that ran (`provenance/hashes/correction_code.sha256`).
  - `make_dd.py` builds the D→D targets from the stored channel and noise of each batch.
- `jobs/` holds the four job scripts that ran, with paths replaced by `${RUN_ROOT}` and `${DEEPSC_SHARED_ROOT}`.
- `summarize_pooled.py` is an offline re-summary of the stored fold files. It pools counts across folds instead of averaging per-fold ratios, and it gives the numbers in the paper:
  `python summarize_pooled.py --output <folder with folds/>`

## What is measured

**Stage 1 (oracle cause).** For each cause, is the restoration map able to restore anything?
- Maps compared:
  - Low SNR and spatial correlation: affine, and a context-conditioned MLP with a variance head.
  - Imperfect CSI: affine, a token MLP, and a block matrix (a learned 2×2 correction from Ĥ and block statistics). There is also an oracle matrix W(H)W(Ĥ)⁻¹, which uses the true H; it is an upper bound only and is exactly invertible.
  - Mismatch: affine at R2, affine at R1, and an MLP at R2.
- Metrics: NMSE and cosine at the actuation stage, then BLEU-1, BLEU-4 and token accuracy after the frozen receiver, and the fraction of the gap to nominal that is recovered.

**Stage 1b.** Harm on nominal inputs, and harm when the wrong map is applied.

**Stage 2 (routing).** Uses the diagnostic's held-out probabilities, temperature-scaled with T fitted on the other folds. Variants: none, oracle, hard routing per sentence, hard routing per block, and the soft same-stage rule per sentence and per block.

**Protocol.**
- Everything is trained on the four training seeds.
- Physical maps are trained on channels re-simulated over training-seed symbols. This uses the exact LMMSE of `channels.py`; reconstruction was checked in the smoke run.
- CSI maps see ρ ∈ {0.99, 0.90} only, so 0.95 is a held-out severity. Correlation maps see ρ_s ∈ {0.3, 0.9} only, so 0.6 is held out.
- Each map acts in the coordinates of its own receiver: C for physical causes, D for mismatch.

## Window analysis (already done, from the saved probabilities)

The scripts are in `experiments/window_analysis/`. Results are in `experiments/window_analysis/WINDOW_RESULTS.md`.

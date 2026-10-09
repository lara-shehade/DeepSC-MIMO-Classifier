# Running the pipeline on a Slurm cluster

The scripts were written for the AUB Octopus cluster:
- `module load python/ai-4`;
- one V100 32 GB per GPU job;
- a job time limit of 5 h 50 min.

On another cluster, adjust the module line and the variables below.

## 1. Preflight

```bash
# Run these commands from the deepsc-mimo-diagnosis directory in the complete handoff.
export DEEPSC_SHARED_ROOT="$(cd ../adv_datasets && pwd -P)"
export DEEPSC_RUNS_ROOT=/path/to/a/writable/results/directory
# optional: DEEPSC_GPU_PARTITION, DEEPSC_CPU_PARTITION, DEEPSC_GPU_RESOURCE (default gpu:v100d32q:1)
bash server/launch_pipeline.sh preflight
```

Preflight:
- creates `$DEEPSC_RUNS_ROOT/<run_id>/`;
- copies the code into `<run_id>/code/` and records its hash (`source_snapshot.sha256`);
- submits one preflight job (data, vocabulary and a CUDA forward/backward check).

All later jobs run from the copied code, never from the repository.

## 2. Main job graph

After the preflight job has succeeded, run:

```bash
bash server/launch_pipeline.sh continue /path/to/runs/<run_id>
```

This submits, with `afterok` dependencies, in this order:
1. training of C and D (8 chunks of 10 epochs each, with exact RNG resume);
2. the C/D integrity check;
3. development trace collection;
4. feature extraction, localization and the classical diagnostics;
5. the 65 neural folds;
6. the neural freeze;
7. arXiv collection and holdout evaluation;
8. the final summary.

Each submission is recorded in `<run_id>/jobs.tsv`. Rerunning `continue` submits only the stages that are missing, and a lock prevents two launchers from running at once.

## 3. Classical stage under a 6-hour job limit

`server/extract_and_classical.slurm` runs feature extraction, localization and all 75 classical fits in one job. In this project's run:
- it ran out of time during the classical fits, after extraction and localization had finished;
- three fusion folds also exceeded 5 h 50 min when run as single jobs;
- the final full-data elastic-net fit took about 7 hours.

**General recovery (works for any run).** Once the original classical job has ended in `TIMEOUT` and the arXiv collection is still pending, run:
```bash
bash server/submit_classical_recovery.sh /path/to/runs/<run_id> --check
bash server/submit_classical_recovery.sh /path/to/runs/<run_id> --apply
```
This does three things:
- it submits the 25 classical folds as separate jobs (`server/classical_recovery.slurm`, 2 CPUs, 16 GB, 5 h 50 min each);
- it submits a finalizer that selects the best model and fits the frozen model;
- it re-points the pending arXiv jobs to that finalizer.

It never repeats feature extraction, and a fold that already has a result is never refit.

**If a fold or the finalizer still exceeds the time limit:**
1. **Missing folds:** run each missing fold where a longer limit is available (another partition, or a workstation), using the same command as the fold jobs:
   ```bash
   cd /path/to/runs/<run_id>/code
   python -m server.classical_recovery ../analysis/development_features.csv \
     --output-root ../analysis/classical --mode fold --fold-index <i>
   ```
   Fold index `i` = 5 × stage + seed position:
   - stages are R0, R1, R2, R3, fusion (0 to 4);
   - seeds are 3101 to 3105 (0 to 4).

   So the fusion folds are 20 to 24. Results go to `analysis/classical/folds/<stage>-<seed>.json`.
2. **Finalizer:** run it the same way with `--mode finalize`. In our run, the final fit took about 7 hours on 2 CPUs.
3. **arXiv jobs:** if the finalizer is run as a Slurm job, point the pending arXiv collection job at it:
   ```bash
   scontrol update JobId=<arXiv job> Dependency=afterok:<finalizer job>
   ```

**What happened in the paper's run.**
- **Step 1:** three fusion folds (3103 to 3105) timed out in the general recovery. They were completed with `classical_tail/`, which splits each of them into one job per estimator.
- **Step 2:** the finalizer timed out. It was completed with `classical_freeze_recovery/`, which runs two chained jobs of 1,500 SAGA iterations each with `warm_start`, the same 3,000 iterations in total.

Both folders are kept as a record. They are tied to that run, so they are not general tools:
- **Hard-coded folds:** the tail handles only fusion folds 3103 to 3105.
- **Job ledgers:** both read the job ledgers of that run, including the stage-subset pilot.
- **Refusal:** they refuse to run on any other state.

**Equivalence.**
- **Cross-validation scores:** the split fold jobs give the same results as the single job. `tests/classical_smoke.py` checks this.
- **Frozen model:** the warm-started final fit affects only the frozen classical model, which is used only for the arXiv classical row.

Without a time limit, the whole classical stage is one command:
`python analysis/classical_diagnostics.py <run>/analysis/development_features.csv --output-root <out>`.

## 4. Add-on experiments

Each add-on runs on a finished run (`RUN_ROOT=/path/to/runs/<run_id>`):

| Order | What | Command (from the repository root) |
|---|---|---|
| 1 | Paper extraction (per-sentence quality, fusion probabilities, calibration, window) | `RUN_ROOT=... DEEPSC_SHARED_ROOT=... bash experiments/paper_extraction/launch.sh` |
| 2 | Stage-subset ablation | see `stage_subset/README.md` (`check`, then `pilot`, then `continue`) |
| 3 | Correction (needs the probabilities from step 1) | `RUN_ROOT=... DEEPSC_SHARED_ROOT=... bash experiments/correction/launch.sh` |
| 4 | Offline analyses (no cluster) | stage attribution, window analysis and the CSI check; see the main README |

## Outputs and isolation

- **Output location:** every job writes only inside its run folder. Each add-on writes only inside its own subfolder (`stage_subset_v1/`, `paper_extraction_v1/`, `correction_v1/`, ...).
- **Code copies:** each add-on copies its code into that subfolder with a hash list. The stage-subset and correction jobs check those hashes again before running.
- **No overwriting:**
  - collection, training checkpoints, the classical and neural diagnostics, the holdout evaluation and the final summary refuse to overwrite existing outputs;
  - the add-on launchers use lock files and refuse to resubmit.

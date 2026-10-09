# Receiver-stage subset ablation (development data only)

This is a separate, supplementary analysis. It does not modify the existing
DeepSC-MIMO pipeline, its Slurm jobs, checkpoints, classical results, or arXiv
holdout. It reads the saved development receiver traces (`R0`–`R3`) and the
20 existing TCN singleton-fold reports. It trains the 11 multi-stage subsets
over five leave-one-physical-seed-out folds (55 new fits), then reports all
15 nonempty subsets. Each subset uses one unchanged TCN branch per stage,
concatenated embeddings, and a linear seven-class head. Training uses the
original neural-fold settings: 12 epochs, batch 64, AdamW 3e-4, weight decay
1e-4, seed 5501, and the existing class/severity balancing code.

The four-stage model is the full-data pilot. Its single fold must finish and
produce a valid report before the remaining 54 folds are submitted. This is
an empirical runtime/memory gate, not a guarantee for all later jobs. The
remaining folds run as a Slurm array with at most four simultaneous tasks.
Each fold writes a unique JSON; the summary starts only after the whole array
completes successfully. A failed or uncertain submission is guarded against
automatic duplication.

Run these commands from any directory, with `RUN` set to a finished pipeline
run:

```bash
RUN=/path/to/runs/<run_id>
bash /path/to/deepsc-mimo-diagnosis/stage_subset/launch.sh check "$RUN"
```

`check` submits no jobs. It validates the 20 existing TCN reports and executes
CPU synthetic tests. It can run before the classical recovery is finished.
After `analysis/classical/FROZEN` and `analysis/frozen_neural.FROZEN` exist:

```bash
bash /path/to/deepsc-mimo-diagnosis/stage_subset/launch.sh pilot "$RUN"
```

After that pilot has completed successfully:

```bash
bash /path/to/deepsc-mimo-diagnosis/stage_subset/launch.sh continue "$RUN"
```

No arXiv artifact or final summary is required. The output is
`$RUN/stage_subset_v1/results/stage_subset_report.json`; each fold is under
`$RUN/stage_subset_v1/results/folds/`. The study keeps a source snapshot,
source hashes, a job ledger, and per-job logs. It refuses to run if the
imported original neural-analysis code changes after the snapshot.

Interpret the results as exploratory development-set evidence. More branches
mean more parameters; the report includes parameter counts. Selecting the
best among 15 subsets using these same five folds introduces selection bias.
Do not present the highest observed score as an independent holdout result,
and do not use arXiv to choose or tune a subset.

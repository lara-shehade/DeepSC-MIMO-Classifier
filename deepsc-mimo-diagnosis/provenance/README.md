# Provenance of the reported results

This folder records how the code in this repository relates to the code that produced the paper's results on the cluster. The run used for the paper started on 2026-09-25.

## Hash lists (`hashes/`)

Each cluster job copied its code into the run folder and saved SHA-256 hash lists. To check the repository files against those lists:

```bash
python provenance/check_hashes.py
```

Expected output: `match: 35, edited doc: 1, not in repo: 1, MISMATCH: 0`.
- **Edited doc:** `stage_subset/README.md`, where cluster paths were replaced.
- **Not in repo:** the run's source archive.

| File | Covers |
|---|---|
| `main_pipeline_dependencies.sha256` | `main.py`, `channels.py`, `utils.py`, `collect_profiles.py`, `models/transceiver.py`, `analysis/neural_diagnostics.py`, as used by the run |
| `stage_subset_*.sha256` | `stage_subset/` and the neural code it imports |
| `classical_recovery_sources.sha256`, `classical_tail_*`, `classical_freeze_recovery_*` | the classical recovery code and `analysis/classical_diagnostics.py` |
| `paper_extraction_code.sha256`, `correction_code.sha256` | `experiments/paper_extraction/`, `experiments/correction/` |
| `run_source_snapshot.sha256` | the run's source archive (not included) |

All Python files in this repository are identical to the run's copies. This was also checked file by file against the run's `code/` folder, beyond the lists above.

## What happened during the run

1. **Training, chunk 2:** the second training chunk failed while restoring the RNG state, because a CPU tensor had been loaded onto the GPU.
   - A two-line fix in `main.py` was installed into the run copy (`rng_resume_amendment.tsv`: old and new hash) with `server/install_rng_resume_fix.sh`.
   - Chunk 2 was resubmitted with `server/recover_second_chunk.sh`.
   - The fix changes no model, data or seed logic. `main.py` in this repository is the fixed version.
2. **Launcher:** the first launcher version failed on an empty Bash array before submitting training.
   - The training DAG was submitted with the corrected launcher. `server/launch_pipeline.sh` here is that version.
   - The run's own copy of the launcher, made at preflight, is the older one.
3. **Resources:** memory requests of the pending jobs were lowered in place with `server/retune_pending.sh`. The `.slurm` headers here carry the reduced values. Resource requests do not affect results.
4. **Classical stage:** the stage was completed with the recovery steps in `docs/DEPLOYMENT.md` §3.
   - The final full-data elastic-net fit ran as two warm-started chunks of 1,500 iterations.
   - That frozen model is used only for the arXiv classical result.
5. **Correction:**
   - The first correction submission (`correction_v1_failed_918059`) failed and is not included. The resubmitted code is in `experiments/correction/`.
   - The paper's correction numbers come from `summarize_pooled.py`, an offline re-summary of the stored folds.

## Job ledgers (`job_ledgers/`)

These are the Slurm job IDs of the paper-extraction and correction runs. They are kept only as a historical record and are not needed to run anything.

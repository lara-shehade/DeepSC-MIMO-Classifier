# Classical tail recovery

> **Record of the paper's run, not a general tool.** This script repaired one specific failure in that run. It refuses to run on any other state. For a new run, follow `docs/DEPLOYMENT.md` §3.

This recovery is narrowly scoped to the three missing fusion folds from array
917245: seeds 3103, 3104, and 3105. The 22 completed fold JSON files remain
read-only.

Each missing fold is split into its three original estimators, producing nine
independent part files under `$RUN/classical_tail_v1/parts`. A pilot runs first;
the other eight jobs cannot start unless it succeeds. A final job validates the
parts, atomically publishes the three missing standard fold JSON files, and
runs the unchanged classical final selection/fit.

Only after successful submission does the launcher replace the dependencies of
the pending arXiv collection and stage-subset pilot with the new finalizer. It
verifies both updates before cancelling the obsolete blocked finalizer. It does
not repeat feature extraction or any completed fold.

Run `check` first, then `start` only after it passes:

```bash
RUN=/path/to/runs/<run_id>
bash classical_tail/launch.sh check "$RUN"
bash classical_tail/launch.sh start "$RUN"
```

# Classical frozen-fit recovery

> **Record of the paper's run, not a general tool.** This script repaired one specific failure in that run. It refuses to run on any other state. For a new run, follow `docs/DEPLOYMENT.md` §3.

Use only if the classical-tail finalizer times out during the final full-data
`fusion elasticnet_logistic` fit. All 25 grouped-CV fold reports are preserved.

The original estimator permits 3,000 SAGA iterations. This recovery performs
two successive 1,500-iteration `fit` calls with scikit-learn `warm_start=True`,
persisting the fitted pipeline between jobs. The preprocessing is deterministic
on the identical full balanced data. This changes execution granularity because
the cluster cannot run the observed approximately seven-hour fit in one job;
the selected features, objective, penalty, class weights, data and total
iteration allowance remain unchanged. The execution method is recorded in the
frozen artifact.

`check` never submits. `start` refuses to run until the prior finalizer has a
terminal failure state, submits two chained six-hour jobs, and reconnects the
pending arXiv and stage-subset pilot dependencies to the second chunk.

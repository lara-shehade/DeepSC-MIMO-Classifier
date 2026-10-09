# DeepSC MIMO classifier handoff

This folder is a complete private handoff for running the project from scratch.

| Folder | Contents |
|---|---|
| `deepsc-mimo-diagnosis/` | Code, Slurm jobs, documentation, tests and the stored results of the paper's run |
| `adv_datasets/` | The prepared shared vocabulary (90,746 tokens) and the Europarl, UK Hansard and arXiv pickle files. These are the same files the paper's run used (identical sizes and sample counts), so do not preprocess them again. |

Not included:
- **Trained checkpoints:** the pipeline trains models C and D itself.
- **Receiver traces and the feature table:** these are needed only for `verify/`, the check that skips training. They are available on request.

## 1. Check the installation (CPU, a few minutes, no cluster)

```bash
cd DeepSC_classifier/deepsc-mimo-diagnosis
pip install -r requirements.txt
python -m pytest tests/
python -m tests.integration_smoke
python -m tests.classical_smoke
```

## 2. Run the pipeline on the cluster

```bash
cd DeepSC_classifier/deepsc-mimo-diagnosis
export DEEPSC_SHARED_ROOT="$(cd ../adv_datasets && pwd -P)"
export DEEPSC_RUNS_ROOT=/path/to/a/writable/results/directory   # needs at least 20 GB free
bash server/launch_pipeline.sh preflight
# after the preflight job has succeeded:
bash server/launch_pipeline.sh continue "$DEEPSC_RUNS_ROOT/<run_id>"
```

Replace only the results path. `docs/DEPLOYMENT.md` covers the rest:
- the full job order;
- what to do if the classical stage exceeds the 6-hour job limit, which it did in the paper's run;
- the add-on experiments (paper extraction, stage subsets, correction).

On a cluster other than AUB Octopus, also check the Slurm partition, the GPU resource and the `module load python/ai-4` line.

**Scale:** in the paper's run, the main job graph took about four days from `continue` to the final summary, including the classical repairs. The run folder takes about 7 GB.

## 3. Compare with the paper

Your outputs can be compared with:
- `deepsc-mimo-diagnosis/results/`, which holds the stored outputs of the paper's run;
- `deepsc-mimo-diagnosis/provenance/README.md`, which explains how this code relates to that run.

Expect close but not bit-identical numbers from a new run, because GPU arithmetic varies across hardware and library versions.

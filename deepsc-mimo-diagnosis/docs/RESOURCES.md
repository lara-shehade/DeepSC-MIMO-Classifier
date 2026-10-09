# Resource requests

`--mem` is **host RAM**, separate from the V100 GPU's 32 GB of device memory.
The older package requested 32 GB host RAM for almost every stage and 64 GB
for classical analysis. Those values were safety margins, not measured peaks.

| Stage | Host RAM now | Reason |
| --- | ---: | --- |
| DeepSC C/D training | 16 GB | 36.96 million parameters; FP32 model, gradients, and Adam states together are about 0.55 GiB, largely on GPU. Pickle data is small. |
| Profile collection, integrity, neural folds/freezing, holdout evaluation | 16 GB | Collection operates one batch at a time. Neural training caches about 600 trace files, so 8 GB is unnecessarily tight without a measured peak. |
| Classical feature extraction and fitting | 32 GB | It holds up to 76,800 rows with roughly 966 feature fields per row plus numerical matrices and model copies. |
| Final summary | 1 GB | Standard-library JSON and streaming SHA-256 only. |

CPU requests stay at 4 for DeepSC training and classical fitting, 2 for other
GPU stages, and 1 for final summary. GPU type, batch size, profiles, seeds,
epoch count, and time limits are unchanged. Prior server collection used 16 GB;
these new values are still estimates, not measured MaxRSS for this exact run.

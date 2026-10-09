# DeepSC 2x2 MIMO unified diagnosis pipeline

This deployment-ready extension preserves the validated 2x2 native-complex
MIMO baseline and adds the approved isolated diagnosis profiles, receiver-stage
recording, grouped classical/neural diagnostics, strict arXiv holdout evaluation,
and a guarded SLURM DAG.

## Frozen common baseline

- Europarl and the shared 90,746-token vocabulary.
- `<PAD>=0`, `<START>=1`, `<END>=2`, `<UNK>=3` (strictly checked).
- 2x2 iid complex Rayleigh MIMO, `H_ij ~ CN(0,1)`.
- Batch-shared fading, matching original DeepSC fading granularity.
- Two complex streams represented by four interleaved real values.
- `1/sqrt(Nt)` total-power scaling.
- System-covariance LMMSE and perfect CSI.
- Original DeepSC-style 5--10 dB training, batch 128, 80 epochs.
- C and D are independent runs with global/channel seeds 42 and 43.
- Epoch-80 `last.pth` is canonical; `best.pth` is a sensitivity checkpoint.

## Approved isolated profiles

| Profile | Definition |
|---|---|
| normal | Europarl, C->C, 10 dB, iid H, perfect CSI |
| low_snr | Same source/H/standard noise draw, scaled to 1 dB |
| high_snr_control | 19 dB control, excluded from seven-way diagnosis |
| imperfect_csi | `Hhat=rho H+sqrt(1-rho^2) Htilde`, rho=.99/.95/.90 |
| spatial_correlation | symmetric Kronecker model, rho=.3/.6/.9 |
| literal | prior semantic-POC 20% deletion/typo/repetition/grammar mixture |
| mismatch | TX-C -> identical physical link -> RX-D |
| ood | frozen C on natural UK Hansard |
| HOLDOUT | arXiv, unavailable until diagnostics are frozen |

Legacy `bad_channel`, `noisy_csi`, and `stale_csi` remain isolated behind
`--allow-legacy-profiles`; the pipeline never uses them.

## Receiver stages

- R0: complex noisy MIMO received sequence `[B,4L,2]`.
- R1: actual equalized channel-decoder input `[B,L,16]`.
- R2: channel-decoder output `[B,L,128]`.
- R3: free-running final semantic-decoder-layer hidden vector immediately
  before vocabulary projection. It includes the state predicting `<END>` and
  stores generated-length masks.

Teacher-forced states, targets, losses, true H, explicit CSI error, labels,
severity, SNR, and domain metadata are never blind-classifier inputs. Audit
metadata is stored separately from R0--R3.

## Reproducibility and leakage controls

- Checkpoints save model, optimizer, epoch, Python/NumPy/Torch/CUDA RNG state,
  and the next deterministic channel realization-set index.
- The original optimizer has no scheduler, so scheduler state is explicitly
  `null`; no new training schedule was invented.
- Training uses eight dependency-chained ten-epoch jobs.
- Physical seeds 3101--3105 define reproducible batch-level draw sequences.
- Paired conditions share source ordering and standardized physical draws.
- Leave-one-physical-seed-out evaluation uses equal class totals in both the
  training and held-out folds.
- CSI/correlation severity sampling is balanced in both training and evaluation.
- arXiv cannot run until classical and neural `FROZEN` markers exist.

See `DEPLOYMENT.md` for the exact two-stage server procedure.

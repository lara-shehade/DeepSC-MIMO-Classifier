# correction_v1 results (5 folds: [3101, 3102, 3103, 3104, 3105])

Sanity: [{"r1_reconstruction_max_abs_err": 9.5367431640625e-07, "r1_reconstruction_low_snr_max_abs_err": 7.152557373046875e-07, "faded_reconstruction_rel_err": 7.251146882936155e-08, "redecode_identical_fraction": 1.0}, {"r1_reconstruction_max_abs_err": 7.152557373046875e-07, "r1_reconstruction_low_snr_max_abs_err": 2.384185791015625e-07, "faded_reconstruction_rel_err": 1.0544720652205797e-07, "redecode_identical_fraction": 1.0}, {"r1_reconstruction_max_abs_err": 1.9073486328125e-06, "r1_reconstruction_low_snr_max_abs_err": 2.384185791015625e-07, "faded_reconstruction_rel_err": 1.0254273519194612e-07, "redecode_identical_fraction": 1.0}, {"r1_reconstruction_max_abs_err": 7.152557373046875e-07, "r1_reconstruction_low_snr_max_abs_err": 4.76837158203125e-07, "faded_reconstruction_rel_err": 1.0276026074507172e-07, "redecode_identical_fraction": 1.0}, {"r1_reconstruction_max_abs_err": 9.5367431640625e-07, "r1_reconstruction_low_snr_max_abs_err": 4.76837158203125e-07, "faded_reconstruction_rel_err": 8.119917538351729e-08, "redecode_identical_fraction": 1.0}]
Temperature per fold: [2.62, 2.5, 2.62, 2.5, 2.5]

## Stage 1: oracle-cause restoration

| Cause | Severity | Map | Stage NMSE | Cosine | BLEU-1 | BLEU-4 | Token acc. | Gap recovered (pooled) |
|---|---|---|---|---|---|---|---|---|
| imperfect_csi | 0.9 | none | 0.486 ± 0.140 | 0.797 ± 0.052 | 0.331 ± 0.067 | 0.135 ± 0.055 | 0.267 ± 0.080 | – |
| imperfect_csi | 0.9 | affine | 0.321 ± 0.105 | 0.816 ± 0.061 | 0.348 ± 0.064 | 0.143 ± 0.056 | 0.271 ± 0.081 | 0.05 |
| imperfect_csi | 0.9 | block_matrix | 0.265 ± 0.060 | 0.846 ± 0.038 | 0.400 ± 0.065 | 0.181 ± 0.066 | 0.332 ± 0.080 | 0.19 |
| imperfect_csi | 0.9 | mlp | 0.244 ± 0.066 | 0.845 ± 0.046 | 0.378 ± 0.073 | 0.162 ± 0.072 | 0.302 ± 0.094 | 0.13 |
| imperfect_csi | 0.9 | oracle_matrix | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.700 ± 0.026 | 0.485 ± 0.034 | 0.694 ± 0.029 | 1.00 |
| imperfect_csi | 0.95 | none | 0.266 ± 0.084 | 0.879 ± 0.037 | 0.444 ± 0.063 | 0.222 ± 0.065 | 0.399 ± 0.072 | – |
| imperfect_csi | 0.95 | affine | 0.194 ± 0.066 | 0.884 ± 0.044 | 0.435 ± 0.063 | 0.214 ± 0.065 | 0.380 ± 0.077 | -0.03 |
| imperfect_csi | 0.95 | block_matrix | 0.150 ± 0.035 | 0.913 ± 0.023 | 0.510 ± 0.048 | 0.276 ± 0.054 | 0.469 ± 0.059 | 0.26 |
| imperfect_csi | 0.95 | mlp | 0.149 ± 0.038 | 0.906 ± 0.029 | 0.477 ± 0.058 | 0.248 ± 0.064 | 0.429 ± 0.070 | 0.13 |
| imperfect_csi | 0.95 | oracle_matrix | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.700 ± 0.026 | 0.485 ± 0.034 | 0.694 ± 0.029 | 1.00 |
| imperfect_csi | 0.99 | none | 0.050 ± 0.016 | 0.976 ± 0.007 | 0.631 ± 0.041 | 0.400 ± 0.050 | 0.616 ± 0.048 | – |
| imperfect_csi | 0.99 | affine | 0.071 ± 0.020 | 0.962 ± 0.010 | 0.580 ± 0.058 | 0.353 ± 0.062 | 0.553 ± 0.071 | -0.74 |
| imperfect_csi | 0.99 | block_matrix | 0.035 ± 0.008 | 0.981 ± 0.005 | 0.666 ± 0.029 | 0.443 ± 0.038 | 0.656 ± 0.034 | 0.51 |
| imperfect_csi | 0.99 | mlp | 0.045 ± 0.014 | 0.974 ± 0.007 | 0.630 ± 0.043 | 0.403 ± 0.053 | 0.615 ± 0.049 | -0.01 |
| imperfect_csi | 0.99 | oracle_matrix | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.700 ± 0.026 | 0.485 ± 0.034 | 0.694 ± 0.029 | 1.00 |
| low_snr | None | none | 0.152 ± 0.011 | 0.927 ± 0.006 | 0.406 ± 0.046 | 0.166 ± 0.044 | 0.340 ± 0.060 | – |
| low_snr | None | affine | 0.076 ± 0.012 | 0.956 ± 0.006 | 0.412 ± 0.044 | 0.164 ± 0.040 | 0.350 ± 0.058 | 0.02 |
| low_snr | None | mlp | 0.055 ± 0.008 | 0.965 ± 0.004 | 0.411 ± 0.041 | 0.161 ± 0.040 | 0.349 ± 0.053 | 0.01 |
| mismatch | None | none | 1.901 ± 0.038 | 0.048 ± 0.019 | 0.088 ± 0.004 | 0.010 ± 0.000 | 0.009 ± 0.001 | – |
| mismatch | None | affine_R1 | 0.676 ± 0.021 | 0.571 ± 0.016 | 0.177 ± 0.006 | 0.021 ± 0.001 | 0.036 ± 0.003 | 0.14 |
| mismatch | None | affine_R2 | 0.512 ± 0.017 | 0.721 ± 0.013 | 0.109 ± 0.003 | 0.014 ± 0.001 | 0.021 ± 0.002 | 0.03 |
| mismatch | None | mlp_R2 | 0.152 ± 0.027 | 0.918 ± 0.016 | 0.181 ± 0.007 | 0.023 ± 0.001 | 0.069 ± 0.009 | 0.14 |
| spatial_correlation | 0.3 | none | 0.009 ± 0.003 | 0.994 ± 0.001 | 0.690 ± 0.024 | 0.472 ± 0.031 | 0.683 ± 0.027 | – |
| spatial_correlation | 0.3 | affine | 0.043 ± 0.014 | 0.976 ± 0.007 | 0.670 ± 0.029 | 0.449 ± 0.037 | 0.658 ± 0.032 | -2.15 |
| spatial_correlation | 0.3 | mlp | 0.024 ± 0.023 | 0.989 ± 0.009 | 0.697 ± 0.017 | 0.480 ± 0.023 | 0.691 ± 0.018 | 0.70 |
| spatial_correlation | 0.6 | none | 0.062 ± 0.016 | 0.965 ± 0.008 | 0.654 ± 0.014 | 0.429 ± 0.017 | 0.642 ± 0.016 | – |
| spatial_correlation | 0.6 | affine | 0.048 ± 0.014 | 0.969 ± 0.007 | 0.662 ± 0.027 | 0.439 ± 0.034 | 0.649 ± 0.031 | 0.17 |
| spatial_correlation | 0.6 | mlp | 0.046 ± 0.023 | 0.974 ± 0.010 | 0.677 ± 0.011 | 0.454 ± 0.015 | 0.668 ± 0.013 | 0.49 |
| spatial_correlation | 0.9 | none | 0.341 ± 0.023 | 0.809 ± 0.014 | 0.611 ± 0.046 | 0.385 ± 0.052 | 0.588 ± 0.054 | – |
| spatial_correlation | 0.9 | affine | 0.143 ± 0.016 | 0.914 ± 0.013 | 0.637 ± 0.044 | 0.420 ± 0.048 | 0.620 ± 0.051 | 0.30 |
| spatial_correlation | 0.9 | mlp | 0.112 ± 0.026 | 0.926 ± 0.014 | 0.658 ± 0.044 | 0.440 ± 0.051 | 0.644 ± 0.052 | 0.53 |

Nominal reference BLEU-1 (C->C for physical causes, D->D for mismatch) is in summary.json.

## Predicted variance vs actual error (Spearman, per batch)

| Cause | Severity | rho |
|---|---|---|
| low_snr | None | 0.119 ± 0.108 |
| spatial_correlation | 0.3 | -0.082 ± 0.130 |
| spatial_correlation | 0.6 | -0.132 ± 0.126 |
| spatial_correlation | 0.9 | -0.151 ± 0.098 |

## Nominal harm (ΔBLEU-1 when the map is applied to nominal blocks)

| Map | Method | ΔBLEU-1 |
|---|---|---|
| imperfect_csi | affine | -0.070 ± 0.026 |
| imperfect_csi | block_matrix | 0.018 ± 0.011 |
| imperfect_csi | mlp | -0.011 ± 0.006 |
| low_snr | affine | -0.020 ± 0.011 |
| low_snr | mlp | 0.000 ± 0.001 |
| mismatch | affine_R1 | -0.535 ± 0.025 |
| mismatch | affine_R2 | -0.609 ± 0.030 |
| mismatch | mlp_R2 | -0.436 ± 0.015 |
| spatial_correlation | affine | -0.042 ± 0.025 |
| spatial_correlation | mlp | 0.001 ± 0.009 |

## Wrong-map harm (ΔBLEU-1 vs uncorrected; rows = true cause, columns = applied map)

| True \ applied | low_snr | imperfect_csi | spatial_correlation | mismatch |
|---|---|---|---|---|
| low_snr | – | -0.052 ± 0.029 | 0.010 ± 0.016 | -0.399 ± 0.102 |
| imperfect_csi | 0.008 ± 0.005 | – | -0.003 ± 0.012 | -0.412 ± 0.100 |
| spatial_correlation | 0.000 ± 0.003 | 0.030 ± 0.007 | – | -0.623 ± 0.070 |
| mismatch | -0.000 ± 0.003 | -0.000 ± 0.005 | -0.001 ± 0.002 | – |

## Stage 2: routing with the diagnostic (BLEU-1 per profile)

| Profile | none | oracle | hard_sentence | hard_block | soft_sentence | soft_block |
|---|---|---|---|---|---|---|
| normal | 0.700 ± 0.026 | 0.700 ± 0.026 | 0.707 ± 0.020 | 0.706 ± 0.021 | 0.707 ± 0.020 | 0.707 ± 0.021 |
| low_snr | 0.406 ± 0.046 | 0.411 ± 0.041 | 0.408 ± 0.042 | 0.411 ± 0.041 | 0.410 ± 0.041 | 0.412 ± 0.041 |
| imperfect_csi | 0.469 ± 0.056 | 0.525 ± 0.047 | 0.518 ± 0.046 | 0.523 ± 0.047 | 0.517 ± 0.047 | 0.518 ± 0.047 |
| spatial_correlation | 0.652 ± 0.009 | 0.677 ± 0.016 | 0.677 ± 0.015 | 0.677 ± 0.015 | 0.676 ± 0.014 | 0.676 ± 0.014 |
| literal | 0.555 ± 0.018 | 0.555 ± 0.018 | 0.559 ± 0.015 | 0.560 ± 0.015 | 0.559 ± 0.014 | 0.560 ± 0.014 |
| mismatch | 0.088 ± 0.004 | 0.109 ± 0.003 | 0.109 ± 0.003 | 0.109 ± 0.003 | 0.109 ± 0.002 | 0.109 ± 0.002 |
| ood_hansard | 0.601 ± 0.021 | 0.601 ± 0.021 | 0.605 ± 0.019 | 0.603 ± 0.020 | 0.605 ± 0.019 | 0.605 ± 0.019 |

Receiver C regime only. Mismatch (receiver D) is reported in its own row of the table above.

| Routing | Mean ΔBLEU-1, physical profiles (receiver C) | Mean ΔBLEU-1, normal/literal/OOD (harm, receiver C) |
|---|---|---|
| oracle | 0.029 ± 0.010 | 0.000 ± 0.000 |
| hard_sentence | 0.026 ± 0.008 | 0.005 ± 0.004 |
| hard_block | 0.028 ± 0.010 | 0.004 ± 0.004 |
| soft_sentence | 0.026 ± 0.009 | 0.005 ± 0.004 |
| soft_block | 0.027 ± 0.009 | 0.005 ± 0.004 |

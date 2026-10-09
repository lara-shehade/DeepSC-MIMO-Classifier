BLEU: nltk sentence_bleu (BLEU-4 with SmoothingFunction.method1). Free-running decoder output vs clean reference. Mean of per-seed means ± std across physical seeds.

| split | profile | severity | BLEU-1 | BLEU-4 | token acc. | source UNK | unseen in train | cond(H) | seeds |
|---|---|---|---|---|---|---|---|---|---|
| development | high_snr_control |  | 0.773 ± 0.011 | 0.581 ± 0.015 | 0.772 ± 0.011 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | imperfect_csi | 0.9 | 0.331 ± 0.067 | 0.135 ± 0.055 | 0.267 ± 0.080 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | imperfect_csi | 0.95 | 0.444 ± 0.063 | 0.222 ± 0.065 | 0.399 ± 0.072 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | imperfect_csi | 0.99 | 0.631 ± 0.041 | 0.400 ± 0.050 | 0.616 ± 0.048 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | literal | 0.2 | 0.555 ± 0.018 | 0.264 ± 0.017 | 0.298 ± 0.011 | 0.059 ± 0.001 | 0.071 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | low_snr |  | 0.406 ± 0.046 | 0.166 ± 0.044 | 0.340 ± 0.060 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | mismatch |  | 0.088 ± 0.004 | 0.010 ± 0.000 | 0.009 ± 0.001 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | normal |  | 0.700 ± 0.026 | 0.485 ± 0.034 | 0.694 ± 0.029 | 0.000 ± 0.000 | 0.011 ± 0.001 | 4.82 ± 0.99 | 5 |
| development | ood_hansard |  | 0.601 ± 0.021 | 0.331 ± 0.025 | 0.601 ± 0.022 | 0.000 ± 0.000 | 0.054 ± 0.003 | 4.82 ± 0.99 | 5 |
| development | spatial_correlation | 0.3 | 0.690 ± 0.024 | 0.472 ± 0.031 | 0.683 ± 0.027 | 0.000 ± 0.000 | 0.011 ± 0.001 | 5.35 ± 1.39 | 5 |
| development | spatial_correlation | 0.6 | 0.654 ± 0.014 | 0.429 ± 0.017 | 0.642 ± 0.016 | 0.000 ± 0.000 | 0.011 ± 0.001 | 7.65 ± 2.60 | 5 |
| development | spatial_correlation | 0.9 | 0.611 ± 0.046 | 0.385 ± 0.052 | 0.588 ± 0.054 | 0.000 ± 0.000 | 0.011 ± 0.001 | 25.54 ± 11.17 | 5 |
| holdout | ood_arxiv |  | 0.395 ± 0.020 | 0.087 ± 0.009 | 0.379 ± 0.026 | 0.000 ± 0.000 | 0.221 ± 0.003 | 5.27 ± 1.57 | 5 |

# Window and aggregation analysis (from saved per-sentence probabilities, 5 held-out folds)

Source: `paper_extraction_v1/probs/fusion_probs_seed*.csv`, the CNN-fusion diagnostic. Every number is on held-out folds. Any selection (rule, floor, bias) is made by nested leave-one-fold-out, never on the reported fold. Scripts are in `window_analysis/`.

## 1. Within one realization, the aggregation rule barely matters
Combination rules compared (Kittler et al., IEEE TPAMI 1998): sum (current), product (mean log-prob), median, majority vote, trimmed mean.

| k (sentences, one realization) | Sum rule (current) | Best rule, nested |
|---|---|---|
| 1 | 0.702 ± 0.032 | 0.702 |
| 4 | 0.758 ± 0.040 | 0.763 (product) |
| 16 | 0.779 ± 0.049 | 0.785 (vote) |
| 64 | 0.784 ± 0.052 | 0.793 (trimmed) |
| 128 | 0.786 ± 0.054 | 0.793 (trimmed) |

The largest gain over plain averaging is 0.009, so the within-block plateau is not caused by the combination rule. Nested class-bias (decision-threshold) adjustment changes 7-way balanced accuracy by between −0.007 and +0.009. It only trades nominal recall against correlation recall.

## 2. Nominal vs. correlation from one channel draw: diagnostic vs. Bayes limit
**Genie.** Under the Kronecker model used in the simulation, with equal priors and ρ_s ∈ {0.3, 0.6, 0.9} in equal shares, the Bayes-optimal test between iid and correlated fading *given the exact 2×2 channel matrix H* reaches the following binary balanced accuracy:

| Realizations m | Nominal recall | Correlation recall | Binary balanced |
|---|---|---|---|
| 1 | 0.832 | 0.746 | **0.789** |
| 2 | 0.887 | 0.806 | 0.846 |
| 3 | 0.914 | 0.837 | 0.875 |
| 5 | 0.934 | 0.886 | 0.910 |
| 8 | 0.950 | 0.928 | 0.939 |

Separate binary tests per severity, at m = 1:
- ρ_s = 0.3: 0.656
- ρ_s = 0.6: 0.833
- ρ_s = 0.9: 0.989

**The diagnostic on the same binary task.** Nominal and correlation sentences only, deciding on p_corr vs. p_nominal, without access to H:

| k (sentences, one realization) | Nominal recall | Correlation recall | Binary balanced |
|---|---|---|---|
| 1 | 0.749 | 0.735 | 0.742 ± 0.073 |
| 8 | 0.754 | 0.750 | 0.752 ± 0.078 |
| 128 | 0.740 | 0.760 | 0.750 ± 0.083 |

From k = 1, the diagnostic reaches 0.742–0.752, within 0.04 of the genie's 0.789, even though it never sees H. Adding sentences from the same realization doesn't close that gap (0.742 → 0.750).

**What can be claimed.** For this pair of causes, a large part of the within-block confusion is intrinsic. One channel draw is weak evidence about the distribution that generated it, especially for mild correlation. This is a statement about the nominal-vs-correlation pair only. It is not a ceiling on the 7-way accuracy, which is a different task.

## 3. Aggregating across channel realizations
Assume a persistent impairment, i.e. the same cause over several coherence blocks. Pooling predictions from different stored batches (independent channel realizations) gives the following (mean over 200 random subsets of m of the 10 batches per run):

| Sentences per realization × realizations | Total sentences | 7-way balanced acc. |
|---|---|---|
| 8 × 1 | 8 | 0.777 ± 0.046 |
| 1 × 8 | 8 | 0.845 ± 0.094 |
| 32 × 1 | 32 | 0.794 ± 0.057 |
| 8 × 5 | 40 | 0.841 ± 0.095 |

**How to read this.**
- Both conditions use the same number of *distinct* sentences. The sentences are randomly assigned to batches, so sentence content is matched in distribution, but they are not the same sentences.
- What differs between the conditions is everything shared within a batch: the channel draw, the CSI-error draw, and the batch-level power normalization.
- So the gain shows that pooling across independent batches helps. It is consistent with the genie trend in Section 2.
- It does **not** isolate channel diversity alone. A clean causal test replays the *same* sentences through several independent H.
- Each run has only 10 realizations per fold, and the subsets overlap, so the spread is large. Use this as a qualitative result, not a headline number.

## Claims for the paper
1. Alternative within-block aggregation rules give negligible gains over averaging.
2. Under the assumed Kronecker model, one exact channel realization gives limited evidence for iid vs. correlated fading: the Bayes-optimal binary classifier given H reaches 0.789 balanced accuracy, and the diagnostic reaches 0.75 on the same binary task without H.
3. The Bayes limit improves rapidly with independent realizations.
4. Pooling predictions across independently generated batches also improves accuracy. These batches differ in channel draw and in their (randomly assigned) sentences, so channel diversity is not isolated.

**Next experiment, for the causal version of claim 4:** retrain the fusion diagnostic once with its weights saved, then score the same sentence set replayed through K independent channel draws versus one shared draw.

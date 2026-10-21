# Pre-registration: robustness of the E14 findings

Status: **approved 2026-10-21, committed to this public repository before any series was run.** The git history of this file is the timestamp. Changes made after a run starts will be logged at the end of this file with their reason, and results under the original design will be reported as well.

Author: Jaewook Kim.

## Purpose

[CHANGELOG E14](../../CHANGELOG_CR.md#e14-pbo-and-the-permutation-test-re-measured-under-a-pre-registered-design) reported two findings from a single daily pipeline:
- **F1.** PBO, computed with the rank statistic of Bailey et al. (2017) on refitted CPCV splits, is not a test for an edge.
- **F2.** A full-shuffle permutation test runs above its nominal size.

A review of the revised paper (SSRN v3.3) raised objections that this design addresses:
- The signal-free series kept each asset's drift.
- The planted arms had no matched β = 0 control.
- PBO was not computed with canonical CSCV, nor with an in-sample measure taken outside the fitted events.
- There was one model family.
- The holdout was not purged.
- No dependence-preserving permutation was tried.

This study asks whether F1 and F2 survive those changes, and whether a permutation design with controlled size exists. The decision rules below are fixed before any number exists.

## 1. Arms

All arms use daily Binance data with the same SHA-256 digests as E14 (`results/true_pbo/summary.json`).

| Arm | Series | Seeds | Construction |
|---|---|---|---|
| A: demeaned i.i.d. null | BTC, ETH, SOL × 200 | 0–199 | The asset's daily close-to-close returns minus their sample mean, resampled with replacement as in E14 |
| B0: demeaned, β = 0 | BTC × 100 | 1000–1099 | As A, on the seeds of the planted arms |
| B1 and B2: demeaned, β = 0.117 and 0.163 | BTC × 100 each | 1000–1099 | As A, plus the E14 planted mechanism r′(t+1) = r(t+1) + β·σ·z(t). β is **not** recalibrated; mean holdout AUC is reported |
| B0′: E14 base (not demeaned), β = 0 | BTC × 100 | 1000–1099 | Matched control for the E14 planted arms |
| C: demeaned stationary bootstrap null | BTC × 200 | 0–199 | Stationary bootstrap (Politis and Romano 1994) of demeaned returns, mean block length 20 days |

An intraday null is outside this study. If one is run later, it will be pre-registered separately.

## 2. Statistics per series

Unless a row says otherwise, everything is as in E14 (`experiments/true_pbo/`):
- the same features, CUSUM events and triple-barrier labels;
- the base CatBoost model;
- CPCV with N = 6, k = 2 and a 20-event purge;
- per-event Sharpe ratios before costs.

| ID | Statistic | Definition |
|---|---|---|
| S1 | PBO, refitted CPCV (E14 method) | Eq. 1 of the paper over the 15 splits. Each of the 8 CatBoost configurations of E14 is refitted on each split. In-sample = Sharpe ratio on the fitted training events |
| S2 | PBO, validation slice | Same splits and grid. Within each split, the training events in time order are split as follows: the last 20% form a validation slice, and the 20 events before it are dropped. Each configuration is fitted on the rest. In-sample = Sharpe ratio on the validation slice. Out-of-sample = Sharpe ratio on the test groups |
| S3 | PBO, canonical CSCV | Out-of-sample per-event returns per configuration come from purged 6-fold cross-validation: each group is predicted by a model trained on the other five, with a 20-event purge on each side. The T × 8 matrix is cut into S = 16 contiguous row blocks. All C(16, 8) = 12,870 splits into two halves are used, with no refitting. In-sample and out-of-sample = Sharpe ratio on each half. PBO by Eq. 1 |
| S4 | PBO, second model family | As S1 with 8 LightGBM configurations: 100 trees × num_leaves {15, 63} × learning rate {0.03, 0.1} × class weight {none, balanced}, seed 42, one thread |
| T1 | Permutation, full shuffle, unpurged holdout | As E14: 100 permutations of the whole label vector, base model refitted with identical settings, pass if the real holdout AUC exceeds the 95th percentile of the permuted AUCs |
| T2 | Permutation, full shuffle, purged holdout | As T1, but the 20 training events immediately before the holdout are dropped for the real fit and every permuted fit |
| T3 | Permutation, circular shift, purged holdout | As T2, but each of the 100 permutations rolls the whole label vector by an offset drawn without replacement from [⌊0.1n⌋, ⌈0.9n⌉], where n is the number of labelled events. This keeps the dependence between neighbouring labels |
| T4 | Permutation, block shuffle, purged holdout | As T2, but each of the 100 permutations shuffles the order of consecutive blocks of 20 events (the last block may be shorter) |
| G | AUC > 0.55 and net Sharpe > 0 (E14 cost rule) | On the unpurged holdout and on the purged holdout |
| G′ | Net Sharpe > 0, cost at every event | 18 basis points charged at every event, a round trip per one-bar position |

**Reported for every arm:**
- the pass rate of each statistic, with a 95% Wilson interval. PBO is reported at both thresholds, < 0.20 and < 0.50;
- mean PBO with a series-level bootstrap 95% interval;
- the conjunctions of AUC, each permutation variant and each net-Sharpe rule on the same holdout.

No statistic or conjunction will be selected after the results are seen.

## 3. Decision rules

Notation:
- size = pass rate in B0;
- power = pass rate in B2;
- the comparison is a one-sided Fisher exact test of power > size (n = 100 each).

**D1: does PBO detect an edge?** For each variant (V1 = S1, V2 = S2, V3 = S3, V4 = S4) and each threshold (0.20, 0.50):

| Verdict | Condition |
|---|---|
| has power | power − size ≥ 20 percentage points and p < 0.05 |
| no power | power − size < 10 percentage points and p ≥ 0.05 |
| inconclusive | otherwise |

Outcome for PBO, in this order of precedence:
1. **A variant with power exists** if V2, V3 or V4 "has power" at some threshold. Its size in A and C is reported beside it.
2. **Not a test for an edge** if V1 has "no power" at both thresholds and at least two of V2–V4 have "no power" at both thresholds.
3. **Condition-dependent** otherwise.

**D2: can the permutation test's size be controlled?** For each of T2, T3, T4:

| Verdict | Condition |
|---|---|
| size controlled | the pass rate is ≤ 7.5% on each of A-BTC, A-ETH and A-SOL, and the pooled 95% Wilson upper bound over those 600 series is ≤ 9% |
| power kept | the pass rate in B2 is ≥ 50% |

Outcome for permutation:
- **design found:** some variant is size-controlled and keeps power;
- **partial:** some variant is size-controlled but does not keep power;
- **none:** otherwise.

T1 versus T2 isolates the effect of purging the holdout, and T2 versus T3 and T4 isolates the effect of label dependence. These comparisons are reported descriptively.

**Use of the outcomes.** The author will use the two outcomes to decide whether to write a further paper:
- a full measurement paper if both outcomes are informative;
- a shorter paper if one is;
- none if neither is.

Whatever the outcomes, every rate above is published as a new changelog entry, including results that overturn E14.

## 4. Run plan

| Stage | Arms | Statistics | Series | Estimated wall time (2 workers) |
|---|---|---|---|---|
| 1 | A-BTC, B0, B1, B2, B0′, C | G, S1–S4 | 800 | about 3.6 h |
| 2 | A (all three assets), B0, B1, B2 | G, G′, T1–T4 | 900 | about 4.0 h |

The estimate scales E14's measured cost of about 0.08 s per model fit per worker. It is uncertain by about ±50%.

Before Stage 1 the following must pass, or the run stops and the failure is reported:
- **Unit tests:**
  - CSCV returns PBO = 0 when one column dominates on every block.
  - With identical columns, CSCV gives PBO = 1 under averaged ties.
  - A circular shift preserves the lag-1 autocorrelation of the labels.
  - A block shuffle with a single block is the identity.
  - The demeaned return vector has mean 0 to 1e-12.
- **Replication:** S1 and T1 on the non-demeaned E14 base must reproduce the stored E14 per-series values for 5 seeds.
- **Timing:** one series per arm. If the measured time exceeds twice the estimate, the run stops and reports.

## 5. What this study does not change

- The checklist wording.
- The list of results that still stand in E13.
- E14's reported numbers, which stay as published; this study adds to them.

## Change log

(empty)

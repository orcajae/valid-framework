# Pre-registration: how often PBO thresholds pass strategies without an edge, and a null-calibrated test of rank persistence

Status: **approved 2026-10-23 and committed to this public repository before any registered series was run.** The git history of this file is the timestamp. Changes made after a run starts are logged at the end of this file with their reason, and results under the original design are reported as well.

**Pilot numbers seen before this version.** An independent review of the first draft ran a scratch pilot outside the registered cells (IID returns, S = 8, B = 99, T 250 and 1,000, M 4 and 8, 100 to 1,000 series per arm). In that pilot:
- the calibrated p-value passed about 5% of equal-mean sets;
- it passed about 20% of unequal-mean sets with no edge;
- it had about 38% power at an annual Sharpe ratio of 1.0;
- PBO < 0.20 passed about 40% of unequal-mean sets.

After that review, two thresholds changed:
- the per-cell size bound in D3 went from 7.5% to 8.0%, because at n = 500 Monte Carlo error would fail a test of exact size 5% about 10% of the time;
- D1 now subtracts the span expected from Monte Carlo error alone.

No other threshold was changed after the pilot. Thresholds added after the pilot: the D3 Layer 2 bound (8.0%) and D4 (10 and 5 points). The pilot measured neither N3 nor Layer 2.

Author: Jaewook Kim.

## Purpose

CHANGELOG E15 found that canonical CSCV (Bailey et al. 2017) responds to a planted effect: at PBO < 0.20 it passed 13, 26 and 46% of series as the effect grew. It also passed 13.0–15.5% of demeaned series without an effect, and 23% when the drift was kept. That evidence comes from one grid (M = 8), one block count (S = 16), one daily pipeline and one planted mechanism.

Bailey et al. propose reading a PBO threshold as a test level: "In accordance with standard applications of the Neyman-Pearson framework, a customary approach would be to reject models for which PBO is estimated to be greater than 0.05" (Bailey et al., §3.1 in the February 2015 version at davidhbailey.com; the section number in the 2017 journal version is to be checked before citation).

This study measures, over a range of designs, two rates:
- **size:** how often a PBO threshold passes a set of strategies none of which has a positive edge;
- **power:** how often it passes a set that contains one.

It also tests a proposal: a p-value for rank persistence that is calibrated on null matrices built from the data.

It asks four questions:
- **Q1.** Does the pass rate of PBO < τ under equal means depend on the design? The design factors are S, M, T, the correlation between configurations, volatility clustering and serial dependence in returns.
- **Q2.** Does PBO pass sets whose configurations differ in mean return when none has a positive edge? If so, PBO measures how persistent the ranking of the configurations is, not whether any of them has an edge.
- **Q3.** Under H0 "all configurations have the same mean", does the calibrated p-value control size across the 19 designs of §6 Stage 2? With what power does it detect one configuration with an edge? How often does it pass sets with unequal means and no edge?
- **Q4.** Can a set with one positive edge be told apart from a set with the same spread of means and no positive edge? Arm N3 has the means of P2 shifted down by the edge.

The decision rules in §4 are fixed before any registered number exists.

## 1. Layer 1: synthetic return matrices

Each series is a T × M matrix of per-period returns. Each column is a configuration.

**Generator.** r(t, m) = μ_m + σ_m·e(t, m), with σ_m = 0.01 unless an arm says otherwise. The per-period Sharpe ratio is SR = μ/σ, and annual values assume 252 periods per year. ρ is the correlation of a Gaussian copula across the M columns.

**Serial structures:**
- **IID:** e(t, ·) is Gaussian, independent over t, with Pearson correlation ρ.
- **GARCH:**
  - A common stochastic variance h(t) = 0.02 + 0.08·u(t−1)² + 0.90·h(t−1), with unconditional value 1.
  - h is driven by u(t) = √h(t)·η(t), where η is a standardised Student-t(5) shock independent of the column shocks.
  - e(t, m) = √h(t)·ε(t, m), where ε(t, ·) has standardised t(5) marginals joined by a Gaussian copula with correlation ρ.
  - The Pearson correlation of ε is ρ_P(ρ): 0, 0.488 and 0.894 at ρ = 0, 0.5 and 0.9. It is computed by numerical integration and recorded in the run log.
  - Returns have finite variance and infinite fourth moment.
  - h(0) = 1, and the first 500 periods are discarded as burn-in.
- **MA10:** e(t, m) = Σ_{k=0}^{9} ξ(t − k, m)/√10, where ξ(t, ·) is Gaussian, independent over t, with Pearson correlation ρ. Returns are serially correlated up to lag 9.

**Grid.** All combinations of the factors below: 216 cells, numbered c = 0…215 in table order, last factor fastest.

| Factor | Levels |
|---|---|
| T (periods) | 250, 1000, 2500 |
| M (configurations) | 4, 8, 16, 32 |
| S (CSCV blocks) | 8, 16 |
| ρ | 0, 0.5, 0.9 |
| Serial structure | IID, GARCH, MA10 |

**Arms** for every cell:

| Arm (index a) | Means μ_m | Replications j |
|---|---|---|
| N1 (1): no edge, equal means | all 0 | 0…999 |
| N2 (2): no edge, unequal means | configuration m at annual SR −1.0·(m−1)/(M−1); the best configuration has SR 0 | 0…999 |
| N3 (3): no positive edge, the spread of P2 | configuration 1 at annual SR 0, the others at −1.0. N3 series j is P2 series j (same seed) with every μ_m reduced by the edge | 0…499 |
| P1, P2, P3 (4, 5, 6): one edge | configuration 1 at annual SR 0.5, 1.0 or 2.0; the others at 0 | 0…499 |

**Reference cell:** T = 1000, M = 8, S = 16, ρ = 0.5, GARCH.

**Reported, outside the decision rules:** the reference cell with unequal volatilities, σ_m = 0.005·4^((m−1)/(M−1)), so that configuration 1 has the lowest volatility. Arms N1, N2 and P2 are run with j = 0…999, 0…999 and 0…499, and seeds `default_rng([a, 216, j])`.

**Seeds.** Series (a, c, j) uses `numpy.random.default_rng([a, c, j])`, except that N3 series j uses the seed of P2 series j, `[5, c, j]`.

## 2. Layer 2: the ML pipeline of E15

Everything is as in E15 S3 (`experiments/inbox7/`) unless stated otherwise: daily Binance data with the E14 digests; features, CUSUM events and triple-barrier labels; purged 6-fold out-of-sample per-event returns.

**Configurations.** CatBoost, 100 iterations, with:
- depth {4, 6, 8};
- learning rate {0.03, 0.1};
- class weight {none, `auto_class_weights='Balanced'`};
- l2_leaf_reg {3, 10}.

That gives 24 configurations, in that nested order with the first factor slowest. Other settings follow E15, including the CatBoost version and thread settings. l2_leaf_reg = 3 is passed explicitly; it is the CatBoost default.

The full grid is fitted once per series. Each sub-grid is a fixed set of columns of the T × 24 matrix:
- M = 4: depth {4, 6} × learning rate {0.03, 0.1}, no class weight, l2 3;
- M = 8: depth {4, 6} × learning rate × class weight, l2 3. This is the E15 grid, in the column order of E15's `GRID_P`;
- M = 16: depth {4, 6} × learning rate × class weight × l2;
- M = 24: the full grid.

**CSCV blocks:** S ∈ {8, 16}.

| Arm | Series | Seeds | Construction |
|---|---|---|---|
| A: demeaned i.i.d. | BTC, ETH, SOL × 200 | 0–199 | As E15 A |
| D: drift kept | BTC × 200 | 0–199 | As E14 (the mean is not removed) |
| C: stationary bootstrap | BTC × 200 | 0–199 | As E15 C |
| B0, B1, B2: planted near a feature | BTC × 100 each | 1000–1099 | As E15 B0, B1 and B2 (β = 0, 0.117 and 0.163, demeaned) |
| F1, F2: planted, partly spanned by the features | BTC × 100 each | 1000–1099 | Demeaned, plus r′(t+1) = r(t+1) + β·σ·z40(t). See below |
| R: real BTC daily | 1 | none | Illustration only; no decision uses it |

**The F1 and F2 signal.**
- z40(t) = (Σ_{s=t−39}^{t} r′(s) − 40μ)/(σ√40) for t ≥ 39, and 0 otherwise. μ and σ are the mean and standard deviation of the bootstrap draw r.
- The series is built recursively as in `make_series` (`experiments/true_pbo/run.py`), with window 40.
- It is stationary only for β < 1/√40 ≈ 0.158.
- Under i.i.d. returns, the features ret_20 and ret_60 explain about 75% of the variance of z40.

**β for F1 and F2.**
- The target is the mean holdout AUC over seeds 2000–2019, meaning the base model's AUC on the unpurged 20% holdout: 0.56 for F1 and 0.60 for F2.
- Grid: β = 0, 0.025, …, 0.150.
- β is linearly interpolated within the first adjacent pair of grid points whose mean AUCs bracket the target.
- If no pair brackets the target by 0.150, the arm is dropped and this is reported.
- Both β values are added to this file's Change log and pushed to the public repository before any of seeds 1000–1099 is run.

## 3. Statistics

**PBO.** Canonical CSCV with E15's `cscv_matrices` and `pbo_fast` (`experiments/inbox7/run.py`), unchanged:
- S contiguous row blocks, whose sizes differ by at most one;
- all C(S, S/2) splits into two halves;
- the Sharpe ratio of a half is mean/(sd + 1e-10), with sd computed with divisor n;
- the in-sample best is chosen by `np.argmax`, so a tie goes to the first-listed configuration;
- out-of-sample ranks of tied configurations are averaged (scipy `rankdata`);
- ω̄ = rank/(M + 1), and PBO is the share of splits with logit(ω̄) ≤ 0.

PBO is also reported under the last-listed and mean-rank tie rules of `experiments/inbox7/ties.py`, with the share of splits that have a tie.

**Threshold screens.** PBO < τ for τ ∈ {0.05, 0.10, 0.20, 0.50}.

**Calibrated p-value for rank persistence (the proposal under test).** It tests H0: all configurations have the same mean. It does **not** test H0: no configuration has a positive edge.
1. Subtract its sample mean from each column of the observed matrix.
2. Resample the rows of the demeaned matrix B = 199 times with a stationary bootstrap. Block lengths are geometric with mean L = 20, and rows wrap around the end. **The resampled matrices are not demeaned again.**
3. Compute PBO on each resample.
4. p = (1 + #{b : PBO*_b ≤ PBO_obs}) / (B + 1).
5. The screen passes if p ≤ 0.05.

Seeds for resample b:
- Layer 1, series (a, c, j): `default_rng([7, a, c, j, b])`. N3 series j uses the stream of its paired P2 series, `[7, 5, c, j, b]`, so that the D4 comparison is exactly paired for the calibrated p-value as well.
- Layer 2: `default_rng([8, k, s, b])`, where s is the series seed and k is the arm index: A-BTC 1, A-ETH 2, A-SOL 3, D 4, C 5, B0 6, B1 7, B2 8, F1 9, F2 10, R 11 (with s = 0).

**Comparator, reported outside the decision rules.** A stationary-bootstrap Reality Check (White 2000), on the same matrices with the same B, L and resampling streams:
- V_obs = max_m √T·mean_m;
- V*_b = max_m √T·(mean*_{b,m} − mean_m), on the resamples of the original (not demeaned) matrix with the same stream;
- p_RC = (1 + #{b : V*_b ≥ V_obs})/(B + 1);
- pass: p_RC ≤ 0.05.

**Block-length sensitivity, reported.** The calibrated p-value with L ∈ {5, 50}, for arms N1 and P2 in the reference cell, and for Layer 2 arm A-BTC at (M = 8, S = 16).

**Reported per series:**
- Layer 1: the population annual Sharpe ratio of the configuration with the highest full-sample Sharpe ratio;
- Layer 2: that configuration's full-sample per-event Sharpe ratio, and the median pairwise correlation of the M columns.

**Reported per cell and arm:**
- the pass rate of each screen, with a 95% Wilson interval;
- mean PBO;
- the mean of p.

## 4. Decision rules

These are fixed before any registered run, and every term in them is defined here.

**D1: are PBO thresholds test levels? (Q1)** For τ = 0.20 and τ = 0.05 separately:
- *span* is the largest N1 pass rate minus the smallest, over the 216 Layer 1 cells;
- *R95* is the 95th percentile of the span of 216 independent Binomial(1000, p̄)/1000 rates, where p̄ is the pooled N1 rate at that τ (10,000 draws, `default_rng(0)`).

| Verdict | Condition |
|---|---|
| design-dependent | span − R95 > 5 percentage points |
| design-invariant | span − R95 ≤ 2 percentage points |
| intermediate | otherwise |

**D2: does PBO respond to unequal means without an edge? (Q2)** In the reference cell, compare the N2 and N1 pass rates of PBO < 0.20 with a one-sided Fisher exact test of N2 > N1 (n = 1,000 each). Call its p-value p_F.

| Verdict | Condition |
|---|---|
| responds | N2 − N1 ≥ 10 percentage points and p_F < 0.05 |
| does not respond | N2 − N1 < 3 percentage points |
| unclear | otherwise |

**D3: does the calibrated p-value control size under equal means, and does it have power? (Q3)** In D3, *size* is the N1 pass rate at p ≤ 0.05.

Size is **controlled** if all three of the following hold:
- size ≤ 8.0% in each of the 19 Layer 1 cells of §6 Stage 2;
- the pooled 95% Wilson upper bound is ≤ 6.5%, where *pooled* means total N1 passes / total N1 series over those 19 cells;
- in Layer 2 arm A (BTC, ETH and SOL pooled, 600 series), the pass rate at p ≤ 0.05 is ≤ 8.0% at both (M = 8, S = 16) and (M = 16, S = 8).

There is **power** if the P2 pass rate in the reference cell is ≥ 50%.

| Verdict | Condition |
|---|---|
| works | size controlled and power |
| size only | size controlled, power < 50% |
| fails | size not controlled |

The N2 and N3 pass rates of the calibrated p-value are reported beside D3. They are the rates at which it passes sets without a positive edge. They are not part of the verdict, because the test is not built to reject them.

**D4: edge versus spread. (Q4)** In the reference cell, compare the P2 and N3 pass rates of PBO < 0.20 on the 500 pairs (N3 series j is P2 series j with the edge removed) with an exact two-sided McNemar test. Call its p-value p_Q4.

| Verdict | Condition |
|---|---|
| PBO separates edge from spread | P2 − N3 ≥ 10 percentage points and p_Q4 < 0.05 |
| PBO does not separate them | \|P2 − N3\| < 5 percentage points |
| unclear | otherwise, including N3 − P2 ≥ 10 percentage points, which is reported as such |

The same comparison is reported for the calibrated p-value.

**D5: Layer 2 against Layer 1 (descriptive).**
- For PBO < 0.20 and for the calibrated p-value, the pass rates of every Layer 2 arm at each (M, S) are reported with Wilson intervals.
- Beside them are the Layer 1 rates of the nearest cells: T = 1000 and ρ = 0.5, under each serial structure. M = 24 is shown next to M = 16 and M = 32.
- Arm D is expected to pass more often than arm A, because drift gives the configurations unequal means.
- No verdict is attached.

**Use of the outcomes.** The author will write at most one paper, and its type depends only on D1–D4. Read the rows top to bottom; the first row whose condition holds applies.

| Condition | Paper |
|---|---|
| D3 = works | A methods paper: a calibrated test of rank persistence under H0 "equal means"; its size under equal means and its power; its N2 and N3 pass rates, stated as the rates at which it passes sets without a positive edge; the size of fixed thresholds (D1); D4; and the Reality Check's size and power on the same series |
| D1 = design-dependent for τ = 0.20 or τ = 0.05 | A short paper on the size and power of PBO thresholds. D2 is presented as the mechanism only if D2 = responds; otherwise its verdict is reported without a mechanism claim. D4 is reported |
| D1 = design-invariant for both τ | A short paper giving a threshold table for practitioners, with D2 and D4 |
| otherwise | No paper. The results are published as a changelog entry only |

Whatever the outcomes, every rate is published as a new changelog entry.

## 5. Implementation fixed in advance

- If the Layer 2 pipeline's preparation step returns None for a series, that series counts as not passing every screen, and this is reported.
- The CatBoost version, the thread count and the data digests are recorded in the run log.
- No statistic, screen or conjunction is selected after the results are seen.

## 6. Run plan and compute

| Stage | Content | Estimate (2 workers) |
|---|---|---|
| 1 | Layer 1 threshold screens: 216 cells × (2 × 1,000 + 4 × 500) series, plus the unequal-volatility variant | about 3–5 h |
| 2 | Layer 1 calibrated p-value and Reality Check in 19 cells: the 18 cells T ∈ {250, 1000} × M ∈ {4, 8, 16} × {IID, GARCH, MA10} at S = 8 and ρ = 0.5, plus the reference cell. Arms N1, N2, N3 and P2, series j = 0…499 of Stage 1, B = 199. L sensitivity for N1 and P2 in the reference cell | about 3–5 h |
| 3 | Layer 2: 1,500 series (plus R) × 24 configurations × 6 folds; CSCV at S ∈ {8, 16} for each sub-grid; the calibrated p-value and the Reality Check at (M = 8, S = 16) and (M = 16, S = 8); L sensitivity for A-BTC at (M = 8, S = 16) | about 4–6 h, including depth-8 fits, which are not yet timed |

The estimates are uncertain by ±50%.

**Fallback, fixed now.** If a timing check projects more than twice the upper end of a stage's estimate, B drops to 99 in that stage. The number of series per arm does not change. This is reported.

**Checks before Stage 1.** All of the following must pass. If any fails, the run stops and the failure is reported.

*Unit tests:*
- CSCV gives PBO = 0 when one column dominates every block, and PBO = 1 for identical columns.
- **Generator**, with one fixed seed per serial structure and ρ:
  - at T = 1,000,000, every pairwise Pearson correlation of e is within 0.01 of the target (ρ under IID and MA10, ρ_P(ρ) under GARCH);
  - every column's annual Sharpe ratio is within 0.1 of its target, with M = 8 and arm P2, at T = 1,000,000 under IID and GARCH and T = 10,000,000 under MA10.
- **Demeaning.** Every column mean of the matrix passed to the bootstrap is 0 within 1e-12.
- **No re-centring.** Over 10,000 resamples of one N1 matrix, the resample column means have a standard deviation greater than 0, and their average is within 3 standard errors of 0.
- p lies in [1/(B+1), 1].

*Replication:* Layer 2 at M = 8 on E15 A-BTC seeds 0–4 reproduces two stored results:
- the S3 block arrays `S3_n`, `S3_sum` and `S3_sumsq` (`results/inbox7/stage1_arrays.npz`) within 1e-12;
- the same PBO counts out of 12,870 (2008, 2647, 11034, 8049 and 5809).

*Timing:*
- One series per arm at every (M, S) in Stage 1, every Stage 2 cell, and one series per Layer 2 arm with all 24 configurations.
- The limit is twice the upper end of each stage's estimate. If Stage 1 exceeds it, the run stops and reports.

Every stage runs under the unattended-run rules of this project:
- deterministic seeds;
- resumable per arm;
- start time, expected end and stop conditions logged;
- private commit of the outputs.

## 7. What this study does not change, and what it does not test

- Not changed: the checklist wording, and the published numbers of E14 and E15.
- Not tested: intraday data, and model families other than CatBoost in Layer 2.
- The calibrated p-value does not test whether any configuration has a positive edge. D4 and the N3 arm measure how far PBO can be read that way.

## Change log

- 2026-10-23, before any registered series was run. Implementation details that the text above leaves open are listed in `run.py` (IMPLEMENTATION_NOTES). These touch the text above:
  - §1 gives ρ_P(0.5) as 0.488. Numerical integration gives 0.4874, and ρ_P(0.9) = 0.8941. The generator test uses the integrated values, which every run log records.
  - §2: F1 and F2 run with the interpolated value rounded to three decimals; the unrounded value is stored beside it. In the calibration, a seed whose preparation step returns None is skipped, and the number of seeds used is recorded.
  - §3: in step 4 of the calibrated p-value, PBO_obs is the PBO of the observed matrix before demeaning. The L = 5 and L = 50 runs reuse the random draws of L = 20 (the same start, uniforms and jump indices); only the block-start rule changes. The Reality Check is computed at L = 20 only.
  - §6: timing checks use seeds outside the registered ranges. Stage 1 starts only after all three stages have been timed, and a stage's B is fixed when its first shard starts. For Stage 3 the B decision uses the serial projection rather than half of it, because each CatBoost fit already uses all cores. Layer 1 uses one BLAS thread per worker; the unit tests show bitwise-identical outputs.
- 2026-10-23 β F1 = 0.058 (interpolated 0.05760; target mean holdout AUC 0.56; seeds 2000–2019)
- 2026-10-23 β F2 = 0.077 (interpolated 0.07721; target mean holdout AUC 0.60; seeds 2000–2019)

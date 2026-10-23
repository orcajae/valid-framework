# Pre-registration: PBO with its companion statistic, and CSCV with an embargo (follow-up to CHANGELOG E17)

Status: **committed to this public repository before any registered series was run.** The git history of this file is the timestamp. Changes made after a run starts are logged at the end of this file with their reason, and results under the original design are reported as well.

**This is a follow-up designed after the results of E17 were known.** E17 (`experiments/cscv_calibration/`, commit 69b6b2a) is the source of both parts:
- In its reference design, PBO < 0.20 passed a set with one configuration at an annual Sharpe ratio of 1.0 about as often as the same set with the edge removed (69.6% and 70.2%).
- Outside its decision rules, E17 reported that equal-mean sets passed PBO < 0.20 more often under MA10 dependence, mainly at T = 250 (15.2%, against 10.1% at T = 2,500).
- After E17 was published, the author computed the following from E17's `stage1_rows.csv.gz` (arm N1, PBO = k_first/K, pooled over M and ρ within each T and S):
  - under MA10 at T = 250 and S = 16, the mean PBO was 0.421 and 17.6% passed at 0.20;
  - under MA10 at T = 2,500 and S = 16, the mean was 0.495 and 9.9% passed;
  - under IID and GARCH the pooled means were 0.498–0.502 at every T and S (cell means 0.486–0.517).

  This is consistent with dependence crossing the boundaries between adjacent CSCV blocks.

**What was computed before this file was committed.**
- No pilot was run. The computations were:
  - the E17 figures above, computed from E17's published rows;
  - the unit tests of §5 without the Layer 2 test. They recompute 300 published E17 PBO counts and run the other checks on development seeds;
  - run-time measurements on development seeds;
  - the decision code, run on random counts not drawn from the generator.
- No probability of loss or embargoed statistic was printed, stored or inspected for any E17 series or any series registered here.
  - In earlier drafts, the generator identity test also formed the loss count of the 300 E17 series. It was held in memory only, and neither printed nor stored.
  - The Layer 2 test was not run before this file was committed.
  - Canonical probabilities of loss can be computed by anyone from E17's public files: Layer 1 series from their seeds, and Layer 2 from `stage3_blockstats.npz`. So this statement cannot be verified from the repository. The new seeds draw from the same distributions as E17's.
- On E17's series, the F4 statistic below is excess₀ = 5.575 percentage points (1,338 of 24,000 series: MA10 3,659 passes, IID 2,321). F4's thresholds were set knowing this.
- F1 uses the thresholds of E17's D4.
- Two independent reviews and a regression check read earlier drafts. None of them ran a registered seed or read E17's Layer 2 matrices. Together they ran:
  - the unit tests (development seeds, and E17's seeds for the 300 published PBO counts), the self-test and the timing check (development seeds);
  - the decision code on fabricated counts, including exact boundary ties;
  - the Layer 2 code on fabricated matrices, with a fabricated arm R row;
  - the gate and run-lock tests in throwaway clones;
  - multiprocessing tests on development seeds, one with a forced stop inside a worker;
  - analytic calculations.

  The design changed after each review:
  - **First review.** PoL < π is close to a one-sided test of the selected configuration's mean (§4). The earlier draft's primary screen was C(0.20, 0.20). From that approximation, the review expected N3 to pass it at roughly 10–15% in the reference cell, close to the 10% bound of what is now F3. After the review, the primary π was set to 0.05 and the level check F2 was added. The expectations at π = 0.05 are given in §4.
  - **Second review.** An analytic calculation from the MA10 autocovariance showed that, without an embargo, MA10 dependence alone raises the level of PoL < 0.05 at T = 250 (§4). F2 was then restricted to the 144 IID and GARCH cells, with the MA10 cells reported beside it without a verdict. The same review noted that F4 = "restores" is expected by construction (§4), and tightened the gates of §5.

Author: Jaewook Kim.

## Purpose

**Part A: the companion statistic.** Bailey et al. (2017; Section 3.2 of the February 2015 revised version) report "the proportion of combinations with negative performance" beside PBO, and note that "even if φ ≈ 0, Prob[R̄ᶜ_{n*} < 0] could be high, in which case the strategy's performance OOS is probably poor for reasons other than overfitting". Part A asks three questions:
- whether a screen that requires both a low PBO and a low probability of loss tells a set with an edge from the same set with the edge removed, which PBO alone did not;
- whether the probability of loss, read as a test, holds its nominal level when the configuration is selected in-sample;
- how often the combined screen passes sets without a positive edge.

**Part B: an embargo between CSCV blocks.** Remove the first e periods of every block, so that rows in different blocks are at least e + 1 periods apart. Part B asks two questions:
- whether this restores the equal-means pass rate of PBO < 0.20 under MA10 dependence in short samples;
- what it costs in power.

## 1. Layer 1: synthetic return matrices

The generator, the 216 cells, the arms, the replication counts and the reference cell are those of E17 (`experiments/cscv_calibration/PREREGISTRATION.md` §1), with new seeds:
- **cells** c = 0…215: T ∈ {250, 1000, 2500} × M ∈ {4, 8, 16, 32} × S ∈ {8, 16} × ρ ∈ {0, 0.5, 0.9} × {IID, GARCH, MA10}, numbered as in E17;
- **arms** N1 and N2 (1,000 series per cell), and N3, P1, P2 and P3 (500 each), with the means of E17. In N1, N2 and N3 the best configuration has mean 0;
- **reference cell** T = 1000, M = 8, S = 16, ρ = 0.5, GARCH (c = 103);
- **seeds:** series (a, c, j) uses `numpy.random.default_rng([18, a, c, j])` with the E17 arm indices a (N1 1, N2 2, N3 3, P1 4, P2 5, P3 6). N3 series j uses the seed of P2 series j, `[18, 5, c, j]`. Development seeds are `[99, 18, a, c, j]`.

The unequal-volatility variant (E17 cell 216) is not run.

## 2. Layer 2: E17's pipeline matrices (secondary analysis of existing data)

Layer 2 reuses the T × 24 per-event out-of-sample return matrices that E17 stored for arms A-BTC, A-ETH, A-SOL, D, C, B0, B1, B2, F1, F2 and R (1,501 series).
- They are resampled exchange data, and for arm R the real series. They stay private.
- E17 published block statistics for every series except R.
- Before any new statistic is computed:
  - every series except R must reproduce E17's published `stage3_blockstats.npz` exactly;
  - arm R must reproduce its eight canonical PBO counts published in E17's `stage3_rows.csv`;
  - the sha256 of the matrix file is recorded.
- Sub-grids M ∈ {4, 8, 16, 24} as in E17; S ∈ {8, 16}.
- The trimmed-block counts, sums and sums of squares of the embargoed statistics are published, so that the embargoed counts can be recomputed.

Layer 2 results are descriptive. No decision rule uses them.

## 3. Statistics

**PBO** is E17's canonical CSCV:
- S contiguous row blocks (`np.array_split`) and all C(S, S/2) splits;
- the Sharpe ratio of a half is mean/(sd + 1e-10), with sd computed with divisor n;
- the in-sample best is chosen by `np.argmax`, so a tie goes to the first-listed configuration;
- out-of-sample ranks of tied configurations are averaged;
- PBO is the share of splits with logit(rank/(M + 1)) ≤ 0.

**Probability of loss (PoL).** In each split, take the out-of-sample Sharpe ratio of the configuration selected in-sample (the configuration whose rank PBO uses). PoL is the share of splits in which that ratio is below 0.

**Embargoed CSCV.** The blocks are those of canonical CSCV. The first e rows are removed from every block before the half statistics are formed. Every split, the selection rule, PBO and PoL are then computed exactly as above from the remaining rows.
- **Layer 1:** e = 9, the number of lags over which MA10 returns are dependent (e(t) and e(t + h) share shocks only when |h| ≤ 9). The same e is applied under every serial structure, so that its cost can be measured where it is not needed.
- **Layer 2:** e = 20 events, the purge length of E15 and E17 (`config.CPCV_PURGE`).
  - Each row is a position (±1) times the next day's return, on distinct event days, so label overlap does not make rows dependent.
  - The embargo separates rows in different blocks by at least 21 events (about 75 days for A-BTC, which averages 873.7 events over 3,132 days) against dependence in returns, features and positions, whose length is not known.
  - e = 20 is a convention, not a derived dependence length. It does not remove dependence shared by all events of one cross-validation fold.

**Screens** (pass = the condition holds):
- P(τ): PBO < τ;
- L(π): PoL < π;
- C(τ, π): PBO < τ and PoL < π;
- the **primary screen** is C(0.20, 0.05);
- each screen is computed without and with the embargo;
- the grid τ, π ∈ {0.05, 0.10, 0.20, 0.50} is reported without a verdict.

**Reported per cell and arm**, without and with the embargo:
- the pass count, n and rate of each screen, with a 95% Wilson interval;
- mean PBO;
- mean PoL.

Layer 2 is reported the same way per arm, sub-grid and block count.

## 4. Decision rules

These are fixed before any registered run. Percentages are compared exactly, as fractions of integer counts.

**What to expect from Part A.**
- Take one configuration with independent rows. Conditional on its full-sample mean, the mean of an out-of-sample half has that mean and a standard deviation of about σ/√T: its deviation from the full-sample mean is half the difference of the two half means, which is independent of the full-sample mean.
  - So PoL ≈ Φ(−t), where t is the configuration's full-sample t-statistic.
  - PoL < π is then close to the one-sided test t > z₁₋π, at level about π.
- F1 is therefore expected to separate by construction. In the reference cell the edge configuration of P2 has a full-sample t-statistic of about 2.0, and the best configuration of N3 has one of about 0.
- At π = 0.05 the same approximation gives:
  - N3 should pass C(0.20, 0.05) at a few percent.
  - P2's edge configuration alone passes L(0.05) about Φ(1.99 − 1.645) ≈ 64% of the time. Selection among the configurations and the conjunction with PBO < 0.20 (69.6% in E17) both lower this.
  - So F3's condition P2 ≥ 50% is the one expected near its bound.
- What is not expected by construction is the level once the configuration has been selected in-sample. F2 measures that level.
- **MA10 without an embargo also moves the level.** From the MA10 autocovariance, the mean correlation r between the in-sample and out-of-sample half means is:
  - 0.121 at T = 250 and S = 16;
  - 0.057 at T = 250 and S = 8;
  - at most 0.027 at T ≥ 1,000;
  - exactly 0 with e = 9.

  Then PoL ≈ Φ(−t·√((1 + r)/(1 − r))), and the level of PoL < 0.05 for one configuration at T = 250 is about 7.3% (S = 16) and 6.0% (S = 8). F2 is therefore evaluated on the IID and GARCH cells, and the MA10 cells are reported beside it.
- π = 0.05 is the conventional one-sided level.
- F3 compares point estimates with no allowance for Monte Carlo error. At n = 500 the standard error is about 1.3 points near 10% and 2.2 points near 50%. F2 allows for it through the pooled Wilson bound and R95.

**F1: does the combined screen separate an edge from a spread of means? (Part A)**
- Reference cell, screen C(0.20, 0.05) without embargo, on the 500 pairs (P2 series j, N3 series j).
- Exact two-sided McNemar test on the discordant pairs; p_M is its p-value. These are the thresholds of E17's D4.

| Verdict | Condition |
|---|---|
| separates | P2 − N3 ≥ 10 percentage points and p_M < 0.05 |
| does not separate | \|P2 − N3\| < 5 percentage points |
| unclear | otherwise |

**F2: does PoL < 0.05 hold its level after selection? (Part A)**
- Screen L(0.05) without embargo, for each of N1, N2 and N3 (in each, the best configuration has mean 0).
- Over the 144 IID and GARCH cells, define:
  - *pooled*: total passes / total series;
  - *span*: largest cell rate minus smallest;
  - *R95*: the 95th percentile of the span of 144 independent Binomial(n, p̄)/n rates, where n is the arm's replication count and p̄ its pooled rate (10,000 draws, `default_rng(0)`, linear interpolation). Span and R95 are computed in integer counts and compared exactly.
- The 6.5% bound is E17's D3 bound on a pooled Wilson upper bound. The 2-point bound is E17's D1 bound for "design-invariant".
- The same quantities for the 72 MA10 cells, without and with the embargo, are reported beside F2 with no verdict.

| Verdict | Condition |
|---|---|
| level holds | for each of N1, N2 and N3, over the 144 IID and GARCH cells: pooled 95% Wilson upper bound ≤ 6.5% and span − R95 ≤ 2 percentage points |
| level fails | otherwise (the arms and conditions that fail are reported) |

**F3: is the combined screen useful in the reference cell? (Part A)**
- Screen C(0.20, 0.05) without embargo.
- The 10.0% bound is E17's PBO < 0.20 pass rate for equal-mean sets in the same cell (9.9%), rounded up. The 50% bound is E17's power criterion.

| Verdict | Condition |
|---|---|
| useful | N1, N2 and N3 each ≤ 10.0%, and P2 ≥ 50% |
| safe but weak | N1, N2 and N3 each ≤ 10.0%, and P2 < 50% |
| not useful | N1, N2 or N3 above 10.0% |

**What to expect from Part B.** With e = 9, rows in different blocks are at least 10 periods apart, so under MA10 the two halves of every split are independent (r = 0 above). The out-of-sample rank of the in-sample best is then uniform in every split, as under IID. Only within-block effects on the spread of PBO across series remain. So F4 = "restores" is expected by construction, and F4 mainly checks the implementation. Part B's main measurement is the cost, F5.

**F4: does the embargo restore the equal-means rate under MA10 in short samples? (Part B)**
- Let r(serial, T, variant) be the pooled N1 pass rate of P(0.20) over the 24 cells with that serial structure and T (M × S × ρ, 24,000 series), for variant ∈ {canonical, embargoed}.
- Let excess₀ = r(MA10, 250, canonical) − r(IID, 250, canonical).
- Let excess_E = r(MA10, 250, embargoed) − r(IID, 250, embargoed).

Read the rows top to bottom; the first that holds applies.

| Verdict | Condition |
|---|---|
| premise not reproduced | excess₀ < 3.0 percentage points |
| restores | \|excess_E\| ≤ 1.0 percentage point |
| overshoots | excess_E < −1.0 percentage point |
| does not restore | excess_E ≥ 3.0 percentage points |
| partial | otherwise |

**F5: what does the embargo cost? (Part B, descriptive, no verdict)** P2 pass rates of P(0.20) and C(0.20, 0.05), canonical and embargoed, are reported in two ways:
- pooled by serial structure × T;
- in the reference cell.

**Use of the outcomes.** This study does not change the type of paper that E17 assigned (a short paper on the size and power of PBO thresholds). The paper reports this study as a pre-registered follow-up designed after E17's results, whatever its outcomes. Its recommendations follow the verdicts:
- **F1 = separates and F2 = level holds:** recommend reporting PoL with PBO, read as a test at about level π. State that the separation was measured in the reference cell only, and add F3's verdict for that cell.
- **F1 = separates and F2 = level fails:** report that PoL separates in the reference cell but does not hold its level after selection under IID and GARCH returns. Recommend measuring its pass rate on matched data without an edge.
- **F1 ≠ separates:** report that the companion statistic does not resolve the problem.
- **F4 = restores:** recommend an embargo of at least the dependence length when returns overlap, with its cost in power (F5).
- **F4 = overshoots, partial, does not restore, or premise not reproduced:** report it as such.

Every rate is published as a new changelog entry (E18).

## 5. Checks before the run

Unit tests, all on development seeds except where stated:
- **Generator identity with E17.** Take 300 rows drawn with `random_state=20261024` from E17's published `results/cscv_calibration/stage1_rows.csv.gz` (cells 0–215). The generator with E17's seeds `[a, c, j]` must reproduce E17's stored PBO count exactly. The identity tests form only the PBO count. No loss count or embargoed statistic is formed on E17's series.
- **CSCV identity.** Without embargo, the PBO count must equal E15's `cscv_matrices` and `pbo_fast` (copied verbatim) on the same matrix, under every serial structure, at S = 8 and 16.
- **Direct loop.** A loop over splits computed from the rows themselves must give the same PBO and loss counts at e = 0 and e = 9.
- **Embargo spacing.** With e = 9, every remaining row of a block must be at least 10 periods after the last row of the previous block.
- **Decision rules.** A self-test on fabricated counts must give the right verdict at the exact boundaries of F1, F2, F3 and F4:
  - F1: 10 points with p_M < 0.05 gives "separates"; exactly 5 points gives "unclear".
  - F2: span − R95 of exactly 2.0 points gives "level holds".
  - F3: exactly 10.0% counts as within the bound.
  - F4: exactly 1.0 points gives "restores"; exactly 3.0 gives "does not restore".
- **Layer 2, run on the private matrices before Layer 2 statistics.** Every series except R must reproduce E17's `stage3_blockstats.npz`, and arm R its eight canonical PBO counts in `stage3_rows.csv`.

**Timing:** projected wall time on development seeds with two workers. The limit is 6 hours.

**Enforced gates.** `layer1` and `layer2` refuse to start unless all of the following hold:
- the unit tests and the timing check have passed with the current `run.py` and the current text of this file before its Change log;
- `run.py` and this file are committed, and identical to `origin/main`;
- no other run is active in the same output folder.

The first launch writes a lock with the sha256 of `run.py` and of this file's text before the Change log. A later launch with different files refuses, and so does the summary.

## 6. Stop conditions

Stop and report, without changing the design, if any of the following happens:
- a unit test fails;
- the timing projection exceeds the limit;
- the Layer 2 identity fails;
- a registered series fails to compute.

The expected end time is not a stop condition.

## 7. What this study does not change, and what it does not test

- **Not changed:** E17's verdicts, its published numbers, and the checklist wording.
- **Not tested:**
  - other embargo lengths;
  - embargoes applied only at boundaries between the two halves;
  - intraday data;
  - other companion statistics (performance degradation, stochastic dominance).
- PoL is computed on Sharpe ratios gross of costs, as PBO is.

## Change log

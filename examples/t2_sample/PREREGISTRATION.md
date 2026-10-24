# Pre-registration: Tier 2 sample review for a fictional client, "Sample Fund A" (synthetic data only)

**What this is.** A demonstration of the deliverable of a Tier 2 review (gate size and power). It is not a study, and none of its numbers are research findings. The client, its pipeline and its data are invented, and every series comes from the fixed-parameter generator in `run.py`. No market data is read.

**How it was registered.** This file and `run.py` were committed to the reviewer's private repository before any registered series was run. In a real review the client receives the commit hash and the sha256 of both files before the run. Here the commit could only be checked from inside the private repository; the hashes are printed in the published report so that the published files can be matched to them. The timing check ran on development replications (numbers 10,000 and up) and printed no pass rate.

Author: Jaewook Kim.

## 1. The client and its pipeline (as the fictional client describes it)

- **Strategy.** One daily long-or-short position in one asset, set at the close and held for the next day.
- **Features at the close of day t.** Trailing 5, 20 and 60-day return sums, each scaled by 0.02 times the square root of the window; and a persistent "flow" feature x.
- **Label.** 1 if the sum of the next five daily returns is positive, else 0. Labels of neighbouring days overlap.
- **Model.** L2 logistic regression with an unpenalised intercept, features standardised on the training rows.
- **Walk-forward.** The first fit uses the first 400 usable days; the model is refitted every 100 days on all earlier usable days whose labels end at or before the close of the first test day; each fit predicts the next 100 days. About 536 test days per series.
- **Configurations (8).** Feature set A (5-day, 20-day, x) or B (20-day, 60-day, x) × regularisation C = 0.1 or 1.0 × trading band 0 or 0.02 (no position while the predicted probability is within the band of 0.5). Otherwise long above 0.5, short below.
- **Costs.** 9 bp per unit change in position, so 18 bp for a round trip in and out.
- **Reported configuration.** The configuration with the highest test-period net Sharpe ratio. This is the backtest the client would show.

## 2. The client's gates (run exactly as defined here)

| Gate | Rule |
|---|---|
| G1 | PBO < 0.20 by canonical CSCV (S = 16 contiguous blocks, all 12,870 splits), on the 8 configurations' test-period daily net P&L. Same code as the E17 study: first argmax in-sample, average out-of-sample ranks, logit ≤ 0 counted |
| G2 | Permutation p < 0.05: the reported configuration's test-period predictions are held fixed, the labels are shuffled 100 times without refitting, and the statistic is AUC; p = (1 + number of shuffled AUCs at or above the observed AUC) / 101 |
| G3 | Test-period AUC of the reported configuration > 0.55 |
| G4 | Test-period net Sharpe ratio of the reported configuration > 0 |
| G-all | All four |

All comparisons are exact (integer counts and fractions).

## 3. Data: four arms, 200 series each, T = 1,000 days

The generator, with every parameter fixed here and not fitted to any data:
- Daily returns r(t) = μ + √h(t) · (b · x(t − 1) + √(1 − b²) · z(t)).
- GARCH(1,1) variance: unconditional daily volatility 0.02, ARCH 0.08, GARCH 0.90. Innovations z are Student-t with 5 degrees of freedom, scaled to unit variance.
- The flow feature x is AR(1) with persistence 0.9 and unit variance, independent of z.
- A 250-day burn-in is dropped.

| Arm | μ (daily) | b | Purpose |
|---|---|---|---|
| N1 | 0 | 0 | No edge, no drift: size |
| N0 | 0.0005 | 0 | No edge, drift kept: size when the asset drifts up |
| P1 | 0 | 0.04 | Planted edge, weaker: power (descriptive) |
| P2 | 0 | 0.08 | Planted edge, stronger: power |

- b is about the daily correlation between x and the next day's return. The two strengths were set as round values before any run. The planted edge works through x, which is one of the client's own features, so power here is the power against an edge the pipeline can see directly.
- **Seeds.** Replication j (0–199) draws z, the flow innovations and x(0) from `numpy.random.default_rng([2026, 1024, j])`. All four arms share these draws, so the arms are paired. The relabelling draws of G2 use `default_rng([2026, 2048, arm index, j])` (arm index in the order N0, N1, P1, P2).

## 4. Decision rules (demonstration thresholds, fixed before the run)

For each gate and for G-all:
- **Size:** N1 passes ≤ 15 of 200 → "size within bound"; otherwise "size exceeds bound". The same rule is reported for N0, labelled "drift kept".
- **Power:** P2 passes ≥ 100 of 200 → "adequate power"; otherwise "weak power". P1 is reported without a verdict.
- Every rate is reported with a 95% Wilson interval.
- These thresholds are for the sample. In a real review they are agreed with the client and written into the pre-registration.

## 5. Alternative constructions and descriptive outputs (no verdict)

- **G2 with circular shifts:** the same as G2, but each relabelling rotates the label series by an offset drawn uniformly from 20 to n − 21 days, which keeps the overlap between neighbouring labels.
- **Costs:** G4 with the reported configuration re-chosen at 0, 9, 18 and 36 bp per round trip.
- Mean PBO, mean AUC of the reported configuration, and how often each configuration is the reported one, per arm.

## 6. What this sample does not show

- Real data. Size and power depend on the data, and a real review builds its no-edge series from the client's own data in more than one way.
- Other pipelines, gates, grids, horizons or cost rules.
- Whether any strategy has an edge in practice.

## Change log

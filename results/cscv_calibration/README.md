# Outputs of the PBO-threshold study (CHANGELOG E17)

Produced on 23 October 2026 by
[`experiments/cscv_calibration/run.py`](../../experiments/cscv_calibration/run.py)
and `summarize.py`, under
[`PREREGISTRATION.md`](../../experiments/cscv_calibration/PREREGISTRATION.md).
The results are described in
[CHANGELOG_CR.md, E17](../../CHANGELOG_CR.md#e17-pbo-thresholds-across-designs-under-a-pre-registered-monte-carlo).

A PBO count `k` is the number of CSCV splits, out of `K`, in which the
in-sample best configuration ranks at or below the out-of-sample median
(logit of rank/(M+1) ≤ 0). PBO = k/K, and a set passes τ if PBO < τ. The
suffixes `_first`, `_last` and `_mean` give the count under three rules for
ties in the in-sample best (first listed, last listed, or the average
out-of-sample rank of the tied configurations);
the decisions use `_first`. `_tie` is the number of splits with a tie.

| File | Contents |
|---|---|
| `decision.json` | D1–D4, the "Use of the outcomes" row, and the numbers behind each, with the sha256 of the pre-registration and of `summarize.py` |
| `stage1_rows.csv.gz` | Layer 1, one row per series: cell, design (T, M, S, ρ, serial structure, unequal volatilities), arm, series index `j`, PBO counts, and the index of the configuration with the highest full-sample Sharpe ratio with its population annual Sharpe ratio. Cells 0–215 are the grid; cell 216 is the unequal-volatility variant of the reference cell |
| `stage2_rows.csv` | Stage 2, the 19 Layer 1 cells where the calibrated p-value was computed: observed count `k_obs`, the number of resamples with a count at or below it (`le_L20`, and `le_L5`, `le_L50` for the block-length check), p-values, and the Reality Check (`ge_rc`, `p_rc`) |
| `stage2_star.npz` | Stage 2, the PBO count of every bootstrap resample, per cell, arm and block length (uint16, series × 199) |
| `stage3_rows.csv` | Layer 2, one row per series: arm, seed, PBO counts for each sub-grid (M = 4, 8, 16, 24) and S ∈ {8, 16}, the calibrated p-values at (M = 8, S = 16) and (M = 16, S = 8), and the Reality Check |
| `stage3_star.npz` | Layer 2, the PBO count of every bootstrap resample (199 per series and setting) |
| `stage3_blockstats.npz` | Layer 2, for each series except arm R: per-block event counts, sums and sums of squares of the 24 configurations' out-of-sample per-event returns, for S = 8 and 16. Every Layer 2 PBO count except arm R's is computed from these |
| `rates_layer1.csv`, `rates_stage2.csv`, `rates_layer2.csv` | Every pass rate with its count, n and 95% Wilson interval |
| `means_layer1.csv` | Layer 1 cell means of PBO under the three tie rules, of the tie share and of the population annual Sharpe ratio of the configuration with the highest full-sample Sharpe ratio |
| `d5_table.csv` | D5 of the pre-registration (descriptive): Layer 2 rates beside the matching Layer 1 cells |
| `beta_F.json` | Calibration of β for arms F1 and F2 on development seeds 2000–2019 |
| `checks/` | Stage locks, timing checks, the unit tests and the E15 replication check, as run |
| `logs/` | One log per process: start and end, expected end, stop conditions, versions, sha256 of the inputs |

**Not published.** The Layer 2 per-event return matrices (`stage3_R.npz`) are
resampled exchange data, and for arm R the real series. `stage3_blockstats.npz`
replaces them for every PBO count except arm R's. The per-shard working files
(`parts/`) are merged into the files above.

**Paths.** Local paths in `checks/` and `logs/` are replaced by `<OUT>`
(this directory) and `<VF>` (the checkout of this repository at 018f4ea that
`run.py` imports). The
sha256 values recorded in the stage locks refer to the files as written
during the run, before this replacement.

**Verify.** From the repository root, with numpy, scipy and pandas:

```bash
python3 -I experiments/cscv_calibration/recompute.py results/cscv_calibration --regen 2000
```

It recomputes the decisions from the rows, Stage 2's observed PBO against
Stage 1, the calibrated-p counts from the stored resample counts, and every
Layer 2 PBO count except arm R's from the block statistics. With `--regen N`
it regenerates N Layer 1 grid series (cells 0–215; N = 2000 draws the same
series as the check reported in E17) from their seeds with a separately
written generator. It prints the number of comparisons of each check and exits
1 on any mismatch or on a check with no comparison. A few minutes on one core.
The calibrated p-values themselves, and arm R's PBO counts, need the per-event
return matrices and are not recomputed.

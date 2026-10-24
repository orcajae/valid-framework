# Outputs of the follow-up to E17: PBO with its probability of loss, and CSCV with an embargo (CHANGELOG E18)

Produced on 24 October 2026 (KST) by
[`experiments/cscv_followup/run.py`](../../experiments/cscv_followup/run.py)
under [`PREREGISTRATION.md`](../../experiments/cscv_followup/PREREGISTRATION.md),
both at commit 801cc0d. The results are described in
[CHANGELOG_CR.md, E18](../../CHANGELOG_CR.md#e18-pbo-with-its-probability-of-loss-and-cscv-with-an-embargo-under-a-pre-registered-follow-up).

For each series and CSCV setting, `K` is the number of splits, `k` the number
of splits in which the in-sample best configuration ranks at or below the
out-of-sample median (logit of rank/(M+1) ≤ 0; ties for the in-sample best go
to the first-listed configuration), and `loss` the number of splits in which
that configuration's out-of-sample Sharpe ratio is below 0. PBO = k/K and
PoL = loss/K. A set passes P(τ) if PBO < τ, L(π) if PoL < π, and C(τ, π) if
both hold. `kE` and `lossE` (Layer 1) and the rows with `e = 20` (Layer 2) are
the same counts with the first e rows of every block removed (e = 9 in
Layer 1, e = 20 events in Layer 2).

| File | Contents |
|---|---|
| `decision.json` | F1–F5 of pre-registration §4 and the numbers behind each, with the sha256 of `run.py`, of the pre-registration and of its text before the Change log, and of the row files |
| `layer1_rows.csv.gz` | Layer 1, one row per series (864,000): cell, design (T, M, S, ρ, serial structure), arm, series index `j`, `K`, `k`, `loss`, `kE`, `lossE` |
| `layer2_rows.csv` | Layer 2, one row per series, sub-grid (M = 4, 8, 16, 24), block count (S = 8, 16) and embargo (e = 0, 20): arm, seed, `K`, `k`, `loss` (24,016 rows, arm R's 16 included) |
| `layer2_blockstats_embargo.npz` | Layer 2, for each series except arm R: per-block event counts, sums and sums of squares of the 24 configurations' out-of-sample per-event returns after the embargo (e = 20), for S = 8 and 16. With E17's `stage3_blockstats.npz` (e = 0), every Layer 2 count except arm R's is computed from these |
| `rates_layer1.csv`, `rates_layer2.csv` | Every pass rate (P, L and C on the grid 0.05, 0.10, 0.20, 0.50) with its count, n and 95% Wilson interval, and the mean PBO and mean PoL, canonical and embargoed |
| `checks/` | The unit tests, the timing check, the run lock, the Layer 2 identity check and the completion records, as run |
| `logs/` | One log per stage: start and end, expected end, stop conditions, versions, sha256 of the inputs |

**Verifier.** `python3 -I experiments/cscv_followup/recompute.py results/cscv_followup results/cscv_calibration --regen 2000`
recomputes the decisions, the rates files and the Layer 2 counts from these
files, and regenerates 3,000 Layer 1 series from their seeds.

**Not published.** The Layer 2 per-event return matrices (E17's
`stage3_R.npz`) are resampled exchange data, and for arm R the real series.
The block statistics replace them for every count except arm R's. The
per-cell working files (`parts/`) are merged into `layer1_rows.csv.gz`.

**Host name.** The machine name in `logs/` is replaced by `<host>`. Nothing
else in `checks/` and `logs/` is changed, and no other file differs from the
run's output.

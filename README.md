# VALID: Validation Architecture for Learning-based Investment Decisions

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- CI badge activates once .github/workflows/ci.yml is live on GitHub:
[![CI](https://github.com/orcajae/valid-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/orcajae/valid-framework/actions) -->

**What it is.** A 12-item validation checklist for financial machine learning —
the tests a trading-strategy backtest must survive before you believe it
(CPCV+PBO, parameter-space variance, permutation tests, cost sensitivity,
baselines, bear-market evaluation). The first domain-specific checklist for
financial ML, analogous to [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378)
for clinical prediction and [REFORMS](https://doi.org/10.1126/sciadv.adk3452)
for ML-based science.

**See it catch an overfit strategy (30 seconds):**

```bash
git clone https://github.com/orcajae/valid-framework.git && cd valid-framework
pip install -e ".[dev]"
python examples/worked_example.py    # mined strategy fails, honest one passes
make reproduce                       # full pipeline: MC nulls, CPCV, DSR, Romano-Wolf
```

`examples/worked_example.py` mines 360 RSI configs the wrong way (selection on
in-sample gross Sharpe), runs them and an honest SMA baseline through all 12
checklist items, and shows the mined "winner" collapsing out-of-sample — with
Romano-Wolf and Deflated-Sharpe leaving 0 of 360 survivors. Everything is
computed live on deterministic synthetic data; see
[REPRODUCE.md](REPRODUCE.md) for the smoke / synthetic / real-data tiers.

**Who made it.** Jaewook Kim (2026), *"Beyond Accuracy: A Validation Framework
for Machine Learning in Cryptocurrency Trading"* —
[SSRN preprint 6508779](https://ssrn.com/abstract=6508779). Accepted for
**oral presentation at the 9th ACM SIGKDD Workshop on Machine Learning in
Finance (KDD-MLF 2026)**, Jeju, August 2026.

## Key Findings

- **340 strategy variants** tested across 3 assets (BTC, ETH, SOL), 4
  timeframes, and 5 model families — 52% (178/340) produce negative net Sharpe
  ratios; only 4.4% (15/340) exceed a simple momentum benchmark
- **Bull bias**: crypto ML models predict 58–97% long without class balancing;
  balancing removes the bias but not the weakness (AUC ≈ 0.50)
- **Statistical-economic disconnect**: a run passing the permutation test
  (p = 0.000, AUC 0.570 vs a shuffled 0.516) still fails CPCV (PBO = 0.267)
  and earns net Sharpe 0.135 against a 0.917 benchmark
- **Cost illusion**: 100% of 15-minute variants negative net of 18 bp; the
  decay is monotone across four frequencies
- **Literature audit**: of 80 papers surveyed (74 empirical, coded), 73%
  ignore class balance, 54% omit transaction costs, 0% use CPCV
- **Monte Carlo false positives**: AUC-based evaluation wrongly passes 27%
  [21%, 34%] of signal-free pipelines; CPCV+PBO cuts this to 0% [0%, 1.9%]

## The VALID Checklist (12 items)

| # | Item | Stage | Failure mode caught |
|---|------|-------|---------------------|
| V1 | Report prediction class distribution | Reporting | Bull bias |
| V2 | Test with/without class balancing | Reporting | Bull bias |
| V3 | Use temporal splitting only | Reporting | Temporal leakage |
| V4 | Apply CPCV with PBO | Reporting | Backtest overfitting |
| V5 | Report parameter-space variance Var(SR_IS) | Reporting | PBO misinterpretation |
| V6 | Permutation tests (≥100 shuffles) | Reporting | Spurious patterns |
| V7 | Net performance with explicit costs | Deployment | Cost illusion |
| V8 | Cost sensitivity analysis | Deployment | Cost illusion |
| V9 | Compare against simple baselines | Reporting | Weak baselines |
| V10 | Evaluate across bear market periods | Deployment | Regime overfitting |
| V11 | Trade frequency and cost-per-alpha | Deployment | Hidden turnover |
| V12 | Provide code for reproducibility | Reporting | Irreproducibility |

The paper's own study self-assesses at **9/12 (3 partial: V5 flatness criteria
disagree, V8 a single cost level, V10 two regimes)** — the checklist is meant
to be applied honestly, including to its authors.

```python
from valid import VALIDChecker

report = VALIDChecker().run_all(
    y_pred_unbal=positions,          # see examples/worked_example.py for a
    y_pred_bal=positions,            # complete, runnable end-to-end usage
    pbo_value=0.15, var_sr_is=0.05,
    gross_sr=1.2, net_sr=0.8, cost_bp=18,
    sr_at_costs={0: 1.2, 18: 0.8, 50: 0.3},
    ml_sr=0.8, baseline_srs={"buy_hold": 0.6, "sma200": 0.7},
    has_temporal_split=True, has_permutation_test=True, permutation_p=0.03,
    has_bear_market_eval=True, trades_per_year=30,
    gross_alpha=1.2, cost_drag=0.4, code_available=True,
)
report.print_summary()
```

## Statistics library

`valid/` implements the underlying statistics as plain, tested functions:

- `cpcv` — combinatorially purged cross-validation (purge + embargo)
- `metrics` — PBO (Bailey & López de Prado), Deflated Sharpe Ratio,
  Var(SR_IS), Wilson and bootstrap CIs
- `multiple_testing` — Bonferroni, Holm, Benjamini-Hochberg, and
  **Romano-Wolf stepdown** with a circular block bootstrap
  (*framework extension beyond the SSRN paper*)
- `labeling` / `costs` — triple-barrier + CUSUM labeling, round-trip cost model

## Reproduce

```bash
make reproduce-smoke   # minutes — CI tier, verifies the pipeline end-to-end
make reproduce         # deterministic synthetic tier, full Monte Carlo
make reproduce-real    # public Binance OHLCV via ccxt (pip install -e ".[data]")
```

Outputs land in `results/` with a run summary (`REPRODUCE_SUMMARY.md`)
comparing against the tracked reference outputs in `results/reference/`.
Tier definitions, measured runtimes, and deviations from the v0.1 reference
run are documented in [REPRODUCE.md](REPRODUCE.md).

## Repository structure

```
valid-framework/
├── valid/                  # the library: checklist, cpcv, metrics,
│                           #   multiple_testing, labeling, costs
├── examples/               # worked_example.py — the 30-second demo
├── experiments/            # reproduction pipeline (see REPRODUCE.md)
├── notebooks/              # worked example + crypto backtesting starter kit
├── tests/                  # pytest suite
├── audit/                  # literature audit (80 papers surveyed, 74 coded)
├── results/reference/      # reference outputs (340-variant corpus, MC nulls)
├── releases/               # downloadable 12-item checklist PDF
└── docker/                 # containerized reproduction
```

## Citation

```bibtex
@article{kim2026valid,
  title={Beyond Accuracy: A Validation Framework for Machine
         Learning in Cryptocurrency Trading},
  author={Kim, Jaewook},
  journal={SSRN Electronic Journal},
  year={2026},
  url={https://ssrn.com/abstract=6508779}
}
```

## For traders

Free, self-contained resources from the same research:

- **[Backtesting checklist (PDF)](releases/backtesting_checklist.pdf)** —
  the 12 items in plain language
- **[Crypto backtesting starter kit (notebook)](notebooks/crypto_backtesting_starter_kit.ipynb)** —
  fetch public data, build an SMA strategy, run the cost-sensitivity analysis
- **[Worked example (notebook)](notebooks/worked_example.ipynb)** — watch the
  checklist catch a data-mined strategy

**VALID Audit** — independent validation of trading-strategy backtests, built
on this framework: contact [@orcajae](https://github.com/orcajae).

*Trading involves risk. Past performance does not guarantee future results.*

## License

MIT — see [LICENSE](LICENSE).

# VALID: Validation Architecture for Learning-based Investment Decisions

> **Correction, October 2026.** Most empirical results of the accompanying paper have been withdrawn after a full re-verification:
> - the literature audit,
> - the CPCV/PBO false-positive result,
> - the multiple-testing survivors,
> - the parameter-space variance comparison.
>
> What was withdrawn, what still stands and why: [CHANGELOG_CR.md, E13](CHANGELOG_CR.md#e13-most-empirical-results-withdrawn-after-a-full-re-verification). The pre-registered re-measurement of PBO and the permutation test: [E14](CHANGELOG_CR.md#e14-pbo-and-the-permutation-test-re-measured-under-a-pre-registered-design). Code-level causes: [KNOWN_ISSUES.md](KNOWN_ISSUES.md). The twelve checklist items are unchanged as a proposal.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- CI badge activates once .github/workflows/ci.yml is live on GitHub:
[![CI](https://github.com/orcajae/valid-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/orcajae/valid-framework/actions) -->

**What it is.** A 12-item validation checklist for financial machine learning —
the tests a trading-strategy backtest must survive before you believe it
(CPCV+PBO, parameter-space variance, permutation tests, cost sensitivity,
baselines, bear-market evaluation). A domain-specific checklist for financial
ML in the spirit of [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) for
clinical prediction and [REFORMS](https://doi.org/10.1126/sciadv.adk3452) for
ML-based science. For the concurrent reporting standard
[QFRS](https://doi.org/10.1007/s10462-026-11664-w) and how the two fit
together, see [CHANGELOG_CR.md, E12](CHANGELOG_CR.md#e12-priority-claims-withdrawn-after-a-concurrent-standard).

**Score your own study in five minutes:**
[orcajae.github.io/valid-framework](https://orcajae.github.io/valid-framework/)
— the 12 items as an interactive scorer, plus the one-page checklist as a PDF.

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
[SSRN preprint 6508779](https://ssrn.com/abstract=6508779), presented orally at
the 9th ACM SIGKDD Workshop on Machine Learning in Finance (KDD-MLF 2026),
Jeju, August 2026. Most of its empirical results are withdrawn (E13).

## What still stands (October 2026)

Each item below was re-verified: the definition in the paper, the computation
in the code and the stored value agree.

- **Net of costs:** 178 of 340 strategy variants (52%) have a negative net
  Sharpe ratio at 18 bp. The share falls as the bars lengthen: 15-minute 9 of 9,
  1-hour 96%, 4-hour 44%, daily 22%.
- **Bull bias:** without class weighting, tree models predict "long" for 90–97%
  of test observations. Weighting removes the bias but not the weakness: AUC
  stays near 0.50.
- **AUC screening on noise:** on signal-free bootstrap series, AUC > 0.55 flags
  27.0% (BTC), 30.0% (ETH) and 26.5% (SOL) as positive.
- **Costs in one run:** in the order-flow 1h run, costs consume 82% of the gross
  Sharpe ratio (0.750 → 0.135).
- **Deflated Sharpe Ratio:** no variant passes once the benchmark is scaled as
  Bailey and López de Prado (2014) define it.

The full list, with what was withdrawn, is in
[E13](CHANGELOG_CR.md#e13-most-empirical-results-withdrawn-after-a-full-re-verification).

## Re-measured under a pre-registered design (E14)

On 600 signal-free series and 200 series with a planted signal:

- **PBO, computed with the paper's rank statistic on refitted CPCV splits,**
  almost never passes at 0.20, with or without a signal (0–1%). At 0.50 it
  passes 31.5% of signal-free BTC series and 8% at the stronger planted signal.
  In-sample selection picks the highest-capacity configurations on almost every
  split, whatever the data. For item V4, report PBO as a description of the
  search, not as a pass-or-fail gate.
- **The permutation test** (100 shuffles) passes 14.0–17.5% of signal-free
  series, about three times its nominal 5%, and 44% and 71% of planted-signal
  series.
- **Item V5** remains unresolved.

Details, design and outputs:
[E14](CHANGELOG_CR.md#e14-pbo-and-the-permutation-test-re-measured-under-a-pre-registered-design),
[`experiments/true_pbo/`](experiments/true_pbo/), [`results/true_pbo/`](results/true_pbo/).

## The VALID Checklist (12 items)

Stage 1 (Statistical) is V1–V6: verifiable from the manuscript alone.
Stage 2 (Economic) is V7–V12: required before anything is traded.

| # | Item | Stage | Failure mode caught |
|---|------|-------|---------------------|
| V1 | Report prediction class distribution | 1 · Statistical | Bull bias |
| V2 | Test with and without class balancing | 1 · Statistical | Bull bias |
| V3 | Use temporal splitting only | 1 · Statistical | Temporal leakage |
| V4 | Apply CPCV with PBO | 1 · Statistical | Backtest overfitting |
| V5 | Report parameter-space variance, Var(SR_IS) | 1 · Statistical | PBO misinterpretation |
| V6 | Include permutation tests (≥100 shuffles) | 1 · Statistical | Spurious patterns |
| V7 | Report net performance with costs | 2 · Economic | Cost illusion |
| V8 | Cost sensitivity analysis | 2 · Economic | Cost illusion |
| V9 | Compare against simple baselines | 2 · Economic | Weak baselines |
| V10 | Evaluate in bear markets and regime transitions | 2 · Economic | Regime overfitting |
| V11 | Report trade frequency | 2 · Economic | Hidden turnover |
| V12 | Provide code for reproducibility | 2 · Economic | Irreproducibility |

The split follows Table 1 of the camera-ready and of the revised SSRN version
(v3.3). Version 3.2 of the SSRN paper grouped V9 and V12 under Stage 1; v3.3
records that change in its Appendix A.

The checklist is meant to be applied honestly, including to its authors. The
paper's own self-assessment was withdrawn in October 2026, because it rested on
results that did not hold up (E13).

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
- `metrics` — PBO (Bailey & López de Prado), Deflated Sharpe Ratio (pass
  `sr_std`, the dispersion of the trial Sharpe ratios, since 0.2.1),
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
The pipeline reproduces the stored outputs. Their `pbo` column is not PBO, and
their Sharpe ratios are scaled by √252 at every bar frequency. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) before reading any number from them.
Tier definitions, measured runtimes, and deviations from the v0.1 reference
run are documented in [REPRODUCE.md](REPRODUCE.md).

## Repository structure

```
valid-framework/
├── valid/                  # the library: checklist, cpcv, metrics,
│                           #   multiple_testing, labeling, costs
├── examples/               # worked_example.py — the 30-second demo
├── experiments/            # reproduction pipeline (see REPRODUCE.md);
│                           #   true_pbo/ = the E14 re-measurement
├── notebooks/              # worked example + crypto backtesting starter kit
├── tests/                  # pytest suite
├── audit/                  # literature audit: withdrawn (see audit/README.md)
├── results/reference/      # reference outputs (340-variant corpus, MC nulls)
├── results/true_pbo/       # E14 outputs (per series, per path, per shuffle)
├── releases/               # downloadable 12-item checklist PDF
├── docker/                 # containerized reproduction
├── KNOWN_ISSUES.md         # code-level definition mismatches (October 2026)
└── CHANGELOG_CR.md         # every correction, E1–E14
```

## Citation

```bibtex
@article{kim2026valid,
  title={Beyond Accuracy: A Validation Framework for Machine
         Learning in Cryptocurrency Trading},
  author={Kim, Jaewook},
  journal={SSRN Electronic Journal},
  year={2026},
  url={https://ssrn.com/abstract=6508779},
  note={Most empirical results withdrawn in October 2026; see CHANGELOG_CR.md, E13 and E14}
}
```

## For traders

Free, self-contained resources from the same research:

- **[Backtesting checklist (PDF)](docs/assets/valid_checklist_onepage.pdf)** —
  the 12 items on one page, each with the question it asks and the scoring rule.
  Built by `scripts/gen_checklist_onepage.py`; the same file is served at
  `releases/backtesting_checklist.pdf` so older links keep working.
- **[Crypto backtesting starter kit (notebook)](notebooks/crypto_backtesting_starter_kit.ipynb)** —
  fetch public data, build an SMA strategy, run the cost-sensitivity analysis
- **[Worked example (notebook)](notebooks/worked_example.ipynb)** — watch the
  checklist catch a data-mined strategy

**VALID Audit** is on hold while the re-verification is completed.

*Trading involves risk. Past performance does not guarantee future results.*

## License

MIT — see [LICENSE](LICENSE).

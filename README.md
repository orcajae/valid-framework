# VALID: Validation Architecture for Learning-based Investment Decisions

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A 12-item validation and reporting framework for financial machine learning research. VALID is the first domain-specific checklist for financial ML, analogous to [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) for clinical prediction and [REFORMS](https://doi.org/10.1126/sciadv.adk3452) for general ML-based science.

## Paper

**"Beyond Accuracy: A Validation Framework for Machine Learning in Cryptocurrency Trading"**

Jaewook Kim (2026). [SSRN preprint](https://ssrn.com/abstract=6508779).

## Key Findings

- **322 strategy variants** tested across 3 assets (BTC, ETH, SOL), 4 timeframes, and 5 model families — 51% produce negative net Sharpe ratios; only 3.7% exceed a simple momentum benchmark
- **Bull bias**: Crypto ML models predict 58–97% long without class balancing; class balancing eliminates bias but does not improve predictive power (AUC ≈ 0.50)
- **Statistical-economic disconnect**: PBO=0.000 + permutation p=0.000, yet net Sharpe = 0.135. First empirical confirmation of Witzany's (2021) PBO critique via 200-iteration Monte Carlo
- **Cost illusion**: ML fails to beat simple momentum even at 0bp transaction costs (SR 0.640 vs 0.954); costs consume 55–91% of gross alpha
- **Literature audit**: 72% of 75 empirical crypto ML papers ignore class balance; 53% omit transaction costs; 0% use CPCV
- **Monte Carlo FPR**: AUC-based evaluation produces 27% [21%, 34%] false positives; CPCV+PBO reduces this to 0% [0%, 1.9%]

## Quick Start

```bash
git clone https://github.com/orcajae/valid-framework.git
cd valid-framework
pip install -e ".[ml,dev]"

# Run the VALID checker on your strategy
python -c "
from valid.checklist import VALIDChecker
checker = VALIDChecker()
report = checker.run_all(
    y_pred_unbal=your_unbalanced_predictions,
    y_pred_bal=your_balanced_predictions,
    pbo_value=0.15,
    var_sr_is=0.05,
    gross_sr=1.2,
    net_sr=0.8,
    cost_bp=18,
    sr_at_costs={0: 1.2, 18: 0.8, 50: 0.3},
    baseline_srs={'buy_hold': 0.6, 'sma200': 0.7},
    has_temporal_split=True,
    has_permutation_test=True,
    permutation_p=0.03,
    has_bear_market_eval=True,
    trades_per_year=30,
    gross_alpha=1.2,
    cost_drag=0.4,
    code_available=True,
)
report.print_summary()
"

# Run tests
make test

# Reproduce all paper results (~2-4 hours)
make reproduce
```

## VALID Checklist (12 Items)

| # | Item | Stage | Failure Mode |
|---|------|-------|-------------|
| V1 | Report prediction class distribution | Reporting | Bull bias |
| V2 | Test with/without class balancing | Reporting | Bull bias |
| V3 | Use temporal splitting only | Reporting | Temporal leakage |
| V4 | Apply CPCV with PBO | Reporting | Backtest overfitting |
| V5 | Report parameter-space variance (Var(SR_IS)) | Reporting | PBO misinterpretation |
| V6 | Include permutation tests (>=100 shuffles) | Reporting | Spurious patterns |
| V7 | Report net performance with explicit costs | Deployment | Cost illusion |
| V8 | Perform cost sensitivity analysis | Deployment | Cost illusion |
| V9 | Compare against simple baselines | Reporting | Weak baselines |
| V10 | Evaluate across bear market periods | Deployment | Regime overfitting |
| V11 | Report trade frequency and cost-per-alpha | Deployment | Hidden turnover |
| V12 | Provide code for reproducibility | Reporting | Irreproducibility |

## Repository Structure

```
valid-framework/
├── valid/                  # VALID Python package
│   ├── checklist.py        # 12-item checker
│   ├── cpcv.py             # CPCV implementation
│   ├── metrics.py          # Var(SR_IS), PBO, DSR, Wilson CI
│   ├── labeling.py         # Triple Barrier + CUSUM
│   ├── features.py         # Feature engineering
│   └── costs.py            # Transaction cost models
├── experiments/            # Reproduce all paper results
├── audit/                  # Literature audit (80 papers, 75 empirical)
├── results/reference/      # Reference outputs (322 variants, MC 200)
├── paper/figures/          # Figures 1-7 (300 DPI)
├── figures/                # Figure generation scripts
├── notebooks/              # Free starter kit (Jupyter)
├── releases/               # Downloadable assets (PDF checklist)
├── tests/                  # Unit tests
└── docker/                 # Docker reproduction environment
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

## For Traders

The VALID framework was built from auditing 80 published crypto trading papers. Here's what we found — and free tools to help you avoid the same mistakes.

### Free Resources

**Backtesting Checklist (PDF)**
12 things to verify before you trust any backtest. Based on our literature audit where 72% of papers had no class balance check and 53% included zero transaction costs.

[Download PDF](https://github.com/orcajae/valid-framework/releases/latest/download/backtesting_checklist.pdf)

**Crypto Backtesting Starter Kit (Jupyter Notebook)**
Complete pipeline: fetch data, build SMA strategy, compute metrics, and run cost sensitivity analysis. The cost analysis in Part 5 is what separates real quants from dreamers.

[Open Notebook](https://github.com/orcajae/valid-framework/blob/main/notebooks/crypto_backtesting_starter_kit.ipynb)

### Key Findings for Practitioners

| What we tested | What we found |
|---|---|
| 340 strategy variants (BTC, ETH, SOL) | Simple momentum beats complex ML after costs |
| Spot vs Perpetual futures | Spot saves 11-15%/yr (funding rate drag) |
| 800 Monte Carlo simulations | Most "alpha" is indistinguishable from noise |
| 7 deep learning architectures | 58-86% bull bias without class balancing |

### Need a Production Bot?

I build automated crypto trading systems with the same realistic cost models used in this research.

- TradingView webhook automation
- Multi-strategy bots (Bybit, Binance)
- Capital Shield risk management
- Every backtest includes cost sensitivity analysis

**[fiverr.com/jwquant](https://fiverr.com/jwquant)** — Message me before ordering.

*Trading involves risk. Past performance does not guarantee future results.*

## License

MIT

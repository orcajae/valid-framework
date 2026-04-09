# VALID: A 12-Item Validation Checklist for Financial Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A 12-item validation and reporting framework for financial machine learning research — the first domain-specific checklist for this field.

## Key Findings

- **340 strategy variants** tested across 3 assets (BTC, ETH, SOL), 4 timeframes, and 8 model families
- **52% produce negative net Sharpe ratios**; only 4.4% exceed a simple momentum benchmark
- **Bull bias**: Tree-based models predict 90–97% long without class balancing; balanced AUC ≈ 0.50
- **Cost illusion**: Transaction costs consume 55–91% of gross ML alpha
- **Monte Carlo FPR**: AUC-based evaluation produces 27% false positives; CPCV+PBO reduces to 0%
- **Multiple testing**: 9 Bonferroni survivors all exhibit PBO=1.0 — standard corrections are necessary but insufficient
- **Literature audit**: 80 papers, median satisfies 2.5 of 12 VALID items; 0% use CPCV

## The 12 VALID Items

| # | Item | Stage | Failure Mode |
|---|------|-------|-------------|
| V1 | Report prediction class distribution | Reporting | Bull bias |
| V2 | Test with/without class balancing | Reporting | Bull bias |
| V3 | Use temporal splitting only | Reporting | Temporal leakage |
| V4 | Apply CPCV with PBO | Reporting | Backtest overfitting |
| V5 | Report Var(SR_IS) | Reporting | PBO misinterpretation |
| V6 | Include permutation tests (≥100) | Reporting | Spurious patterns |
| V7 | Report net performance with costs | Deployment | Cost illusion |
| V8 | Cost sensitivity analysis | Deployment | Cost illusion |
| V9 | Compare against simple baselines | Deployment | Weak baselines |
| V10 | Evaluate across bear markets | Deployment | Regime overfitting |
| V11 | Report trade frequency | Deployment | Hidden turnover |
| V12 | Provide code for reproducibility | Reporting | Irreproducibility |

## Quick Start

```bash
pip install -e ".[ml,dev]"

# Run the VALID checker
from valid.checklist import VALIDChecker
checker = VALIDChecker()
report = checker.run_all(...)
report.print_summary()

# Run tests
make test

# Reproduce all results
make reproduce
```

## Repository Structure

```
valid-framework/
├── valid/                  # VALID Python package
│   ├── checklist.py        # 12-item checker
│   ├── cpcv.py             # CPCV implementation (N=6, k=2)
│   ├── metrics.py          # Var(SR_IS), PBO, DSR, Wilson CI
│   ├── labeling.py         # Triple Barrier + CUSUM
│   ├── features.py         # Feature engineering
│   └── costs.py            # Transaction cost models
├── experiments/            # Reproduce all results
├── audit/                  # Literature audit (80 papers)
├── results/reference/      # Reference outputs (340 variants, MC 800)
├── paper/kdd-mlf/          # Workshop paper (ACM sigconf, 8p)
├── tests/                  # Unit tests (18 pass)
└── docker/                 # Docker reproduction environment
```

## License

MIT

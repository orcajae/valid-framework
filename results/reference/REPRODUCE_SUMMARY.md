# Reproduction Run Summary

- Date: 2026-07-09
- Tier: default
- Platform: macOS-15.6.1-arm64-arm-64bit / Python 3.11.4

## Monte Carlo null-pipeline FPR

| setting | gate | FPR | Wilson 95% CI | n |
|---|---|---|---|---|
| btc_daily | AUC > 0.55 | 31.0% | [25.0%, 37.7%] | 200 |
| btc_daily | Permutation | 21.0% | [15.9%, 27.2%] | 200 |
| btc_daily | PBO < 0.20 | 0.0% | [0.0%, 1.9%] | 200 |
| btc_daily | Net SR > 0 | 54.0% | [47.1%, 60.8%] | 200 |
| btc_daily | Full VALID | 0.0% | [0.0%, 1.9%] | 200 |
| eth_daily | AUC > 0.55 | 32.5% | [26.4%, 39.3%] | 200 |
| eth_daily | Permutation | 19.5% | [14.6%, 25.5%] | 200 |
| eth_daily | PBO < 0.20 | 0.0% | [0.0%, 1.9%] | 200 |
| eth_daily | Net SR > 0 | 40.5% | [33.9%, 47.4%] | 200 |
| eth_daily | Full VALID | 0.0% | [0.0%, 1.9%] | 200 |
| sol_daily | AUC > 0.55 | 30.0% | [24.1%, 36.7%] | 200 |
| sol_daily | Permutation | 17.5% | [12.9%, 23.4%] | 200 |
| sol_daily | PBO < 0.20 | 0.0% | [0.0%, 1.9%] | 200 |
| sol_daily | Net SR > 0 | 51.5% | [44.6%, 58.3%] | 200 |
| sol_daily | Full VALID | 0.0% | [0.0%, 1.9%] | 200 |

### Side-by-side with tracked reference runs

| setting | this run | reference (real data, CatBoost, 200 iter) |
|---|---|---|
| btc_daily | AUC-gate 31.0% / PBO-gate 0.0% (n=200) | AUC-gate 27.0% / PBO-gate 0.0% (n=200) |
| eth_daily | AUC-gate 32.5% / PBO-gate 0.0% (n=200) | AUC-gate 30.0% / PBO-gate 0.0% (n=200) |
| sol_daily | AUC-gate 30.0% / PBO-gate 0.0% (n=200) | AUC-gate 26.5% / PBO-gate 0.0% (n=200) |

Synthetic/smoke tiers verify the *mechanism* (a signal-free pipeline passes naive gates at high rates and CPCV+PBO gates at low rates); the reference numbers come from full real-data runs. See REPRODUCE.md for tier definitions and deviations.

## Multiple-testing survivors

| method | survivors |
|---|---|
| raw p<0.05 | 53/174 |
| Bonferroni | 7/174 |
| Holm | 7/174 |
| Benjamini-Hochberg | 38/174 |
| Romano-Wolf | 8/174 |
| Harvey t>3 | 20/174 |
| DSR>0.95 | 0/174 |

## Worked example (VALID checklist scoreboard)

| strategy | VALID score |
|---|---|
| MinedRSI | 10/12 |
| HonestSMA | 11/12 |

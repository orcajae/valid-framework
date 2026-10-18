# Known issues (October 2026)

Found while re-verifying the paper before journal submission. The paper-level consequences are in [CHANGELOG_CR.md, E13](CHANGELOG_CR.md#e13-most-empirical-results-withdrawn-after-a-full-re-verification). This file lists the code-level causes, so that anyone reusing the code knows what a stored number means.

The stored outputs in `results/reference/` still reproduce from their inputs. What changed is what they can be taken to mean.

| # | Where | What the code computes | What the paper said it computes | Status |
|---|---|---|---|---|
| 1 | `experiments/mc_expansion.py` (the `pbo` variable), and the private corpus scripts that produced the `pbo` column of `variants_340.csv`, `mc_null_catboost_200.csv` and `multi_asset_extended.csv` | The share of CPCV paths in which **one** model's in-sample accuracy exceeds its out-of-sample accuracy. It equals 1.0 in every stored row (1,027 values), and also for models with a planted signal | PBO as defined by Bailey et al. (2017): the out-of-sample rank of the in-sample-best of several configurations | Not PBO. Do not use as a gate. The library function `valid.metrics.compute_pbo` implements the rank version |
| 2 | `experiments/mc_expansion.py` (`var_sr_is`) | Variance across the 15 CPCV paths of **one** model's in-sample Sharpe ratio | Variance across **configurations** (paper Eq. 2). The real-data value 0.012 is a variance across three configurations | The Monte Carlo null and the real value measure different things |
| 3 | `experiments/mc_expansion.py` (permutation test) | 20 label shuffles per series, permutation models trained with 50 boosting iterations against 100 for the real model | V6 asks for at least 100 permutations of the same pipeline | The test is weaker than the checklist's own item |
| 4 | `experiments/mc_expansion.py` (`net_sr`, `gross_sr`, `sr_is`) and the corpus scripts | Per-event returns × √252 at every bar frequency | Annualised Sharpe ratios | The annualisation factor should be √(events per year). Signs are unaffected; magnitudes are not comparable across frequencies |
| 5 | `valid.metrics.deflated_sharpe_ratio` before 0.2.1, and the private script behind `results/reference/multiple_testing.json` | The benchmark was the expected maximum of N standard normals, not scaled by the dispersion of the trial Sharpe ratios. The JSON also turned annualised Sharpe ratios into t-statistics with the square root of the event count | Bailey & López de Prado (2014) | Fixed in 0.2.1: pass `sr_std`. With consistent units, the corpus has no Bonferroni, Holm, BH or t > 3 survivors (largest t 0.76), and still no DSR survivor |
| 6 | `valid/checklist.py` (`pbo_threshold=0.50`) vs the experiments (`PBO < 0.20`) | Two different pass thresholds | One threshold | Unresolved; it will be settled with the pre-registered recomputation |

Reporting a new issue: open a GitHub issue or write to jwim1101@gmail.com.

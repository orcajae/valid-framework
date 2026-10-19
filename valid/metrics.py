"""Statistical metrics for financial ML validation."""
import warnings

import numpy as np
from scipy.stats import norm, rankdata


def annualized_sharpe(returns, periods=252):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods)


def var_sr_is(sr_matrix):
    """Variance of IS Sharpe ratios across configs per fold.
    sr_matrix: (n_configs, n_folds) array of IS SRs.
    Returns mean variance across folds.
    """
    variances = []
    for fi in range(sr_matrix.shape[1]):
        col = sr_matrix[:, fi]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            variances.append(np.var(valid))
    return np.mean(variances) if variances else np.nan


def compute_pbo(is_matrix, oos_matrix):
    """Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu, 2017).

    For each path (column), the configuration with the highest in-sample value
    is located; its out-of-sample rank among the M configurations (1 = worst)
    gives the relative rank w = rank / (M + 1). The path counts as overfit when
    logit(w) <= 0, i.e. w <= 0.5. Returns (pbo, count, total).

    Before 0.2.2 the relative rank was rank / M, which differs from the
    definition when M is odd (for M = 3, a middle rank was not counted).
    """
    n_folds = is_matrix.shape[1]
    pbo_count, pbo_total = 0, 0
    for fi in range(n_folds):
        is_col, oos_col = is_matrix[:, fi], oos_matrix[:, fi]
        valid = ~(np.isnan(is_col) | np.isnan(oos_col))
        if valid.sum() < 2:
            continue
        best_is = np.argmax(is_col[valid])
        ranks = rankdata(oos_col[valid]) / (valid.sum() + 1)
        if ranks[best_is] <= 0.5:
            pbo_count += 1
        pbo_total += 1
    pbo = pbo_count / pbo_total if pbo_total > 0 else np.nan
    return pbo, pbo_count, pbo_total


def deflated_sharpe_ratio(observed_sr, n_strategies, T, skew, kurtosis,
                          sr_std=None):
    """Bailey & Lopez de Prado (2014) DSR.

    All Sharpe ratios must be per-period (not annualised), and T is the number
    of return observations behind ``observed_sr``.

    The benchmark is SR0 = sr_std * E[max of n_strategies standard normals],
    where ``sr_std`` is the cross-sectional standard deviation of the Sharpe
    ratios of all trials, in the same per-period units. Before 0.2.1 the
    expected maximum was used unscaled (``sr_std`` = 1), which is only correct
    if the trial Sharpe ratios have unit dispersion; see KNOWN_ISSUES.md.

    Returns (dsr, sr0, se). ``sr0`` equals the unscaled expected maximum when
    ``sr_std`` is not given.
    """
    euler = 0.5772156649
    ln_n = np.log(max(n_strategies, 2))
    e_max = (np.sqrt(2 * ln_n) * (1 - euler / (2 * ln_n))
             + euler / np.sqrt(2 * ln_n))
    if sr_std is None:
        warnings.warn(
            "deflated_sharpe_ratio: sr_std not given; the benchmark assumes the "
            "trial Sharpe ratios have unit dispersion. Pass the cross-sectional "
            "std of the trial Sharpe ratios (per-period units).",
            stacklevel=2)
        sr_std = 1.0
    sr0 = sr_std * e_max
    se = np.sqrt((1 - skew * observed_sr +
                  (kurtosis - 1) / 4 * observed_sr**2) / T)
    dsr = norm.cdf((observed_sr - sr0) / se) if se > 0 else 0.0
    return dsr, sr0, se


def wilson_ci(p, n, z=1.96):
    """Wilson confidence interval for proportions."""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    a = z / d * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0, c - a), min(1, c + a))


def bootstrap_ci(arr, func=np.mean, n_boot=1000, ci=0.95, seed=None):
    """Bootstrap confidence interval. Deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = rng.choice(arr, size=len(arr), replace=True)
        vals.append(func(s))
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return lo, hi

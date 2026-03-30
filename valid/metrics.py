"""Statistical metrics for financial ML validation."""
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
    """Probability of Backtest Overfitting (Bailey & Lopez de Prado).
    Returns (pbo, count, total).
    """
    n_folds = is_matrix.shape[1]
    pbo_count, pbo_total = 0, 0
    for fi in range(n_folds):
        is_col, oos_col = is_matrix[:, fi], oos_matrix[:, fi]
        valid = ~(np.isnan(is_col) | np.isnan(oos_col))
        if valid.sum() < 2:
            continue
        best_is = np.argmax(is_col[valid])
        ranks = rankdata(oos_col[valid]) / valid.sum()
        if ranks[best_is] <= 0.5:
            pbo_count += 1
        pbo_total += 1
    pbo = pbo_count / pbo_total if pbo_total > 0 else np.nan
    return pbo, pbo_count, pbo_total


def deflated_sharpe_ratio(observed_sr, n_strategies, T, skew, kurtosis):
    """Bailey & Lopez de Prado (2014) DSR."""
    euler = 0.5772156649
    ln_n = np.log(max(n_strategies, 2))
    e_max = (np.sqrt(2 * ln_n) * (1 - euler / (2 * ln_n))
             + euler / np.sqrt(2 * ln_n))
    se = np.sqrt((1 - skew * observed_sr +
                  (kurtosis - 1) / 4 * observed_sr**2) / T)
    dsr = norm.cdf((observed_sr - e_max) / se) if se > 0 else 0.0
    return dsr, e_max, se


def wilson_ci(p, n, z=1.96):
    """Wilson confidence interval for proportions."""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    a = z / d * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0, c - a), min(1, c + a))


def bootstrap_ci(arr, func=np.mean, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval."""
    vals = []
    for _ in range(n_boot):
        s = np.random.choice(arr, size=len(arr), replace=True)
        vals.append(func(s))
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return lo, hi

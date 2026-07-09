"""Multiple-testing corrections for strategy panels.

Given N candidate strategies evaluated on the same sample, naive per-strategy
p-values overstate significance: the more configurations are tried, the more
likely the best one is a fluke. These functions adjust for that selection.

Note: this module is a framework extension beyond the SSRN paper (6508779) —
the paper's five contributions do not include Romano-Wolf; results produced
here are not reproductions of paper numbers.

References
----------
- Lo (2002), "The Statistics of Sharpe Ratios", FAJ 58(4).
- Holm (1979), "A simple sequentially rejective multiple test procedure",
  Scand. J. Statist. 6(2).
- Benjamini & Hochberg (1995), "Controlling the false discovery rate",
  JRSS-B 57(1).
- Romano & Wolf (2005), "Stepwise multiple testing as formalized data
  snooping", Econometrica 73(4).
- Romano & Wolf (2016), "Efficient computation of adjusted p-values for
  resampling-based stepdown multiple testing", Stat. Prob. Letters 113.
- Politis & Romano (1992), circular block bootstrap.
"""
import numpy as np
from scipy.stats import norm


def sharpe_pvalues(returns_matrix, benchmark=0.0):
    """One-sided p-values for H0_i: E[r_i - b] <= 0, one per strategy column.

    Per Lo (2002, iid case): t_i = SR_i * sqrt(T) / sqrt(1 + 0.5 * SR_i^2)
    where SR_i is the per-period Sharpe ratio of excess returns (ddof=1).
    The t statistic is invariant to annualization. p_i = 1 - Phi(t_i).

    returns_matrix: (T, N) array of per-period returns.
    benchmark: scalar or (T,) array subtracted from every column.
    Returns (pvals, t_stats), each shape (N,).
    """
    X = np.asarray(returns_matrix, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    b = np.asarray(benchmark, dtype=float)
    X = X - (b[:, None] if b.ndim == 1 else b)
    T = X.shape[0]
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sr = np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
    t = sr * np.sqrt(T) / np.sqrt(1 + 0.5 * sr**2)
    t = np.where(sd > 0, t, -np.inf)
    pvals = 1 - norm.cdf(t)
    return pvals, t


def bonferroni(pvals, alpha=0.05):
    """Bonferroni FWER control: p_adj = min(1, m * p).

    Returns (reject_mask, p_adj).
    """
    p = np.asarray(pvals, dtype=float)
    p_adj = np.minimum(1.0, len(p) * p)
    return p_adj <= alpha, p_adj


def holm(pvals, alpha=0.05):
    """Holm (1979) step-down FWER control.

    Sort ascending; q_(i) = (m - i + 1) * p_(i); adjusted p is the running
    maximum of q (enforces monotonicity), capped at 1, then unsorted.
    Uniformly more powerful than Bonferroni. Returns (reject_mask, p_adj).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = (m - np.arange(m)) * p[order]
    q = np.minimum(1.0, np.maximum.accumulate(q))
    p_adj = np.empty(m)
    p_adj[order] = q
    return p_adj <= alpha, p_adj


def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg (1995) step-up FDR control.

    Sort ascending; p_adj_(i) = min_{j >= i} min(1, m/j * p_(j))
    (reverse cumulative minimum), then unsorted. Controls the false
    discovery rate, not FWER. Returns (reject_mask, p_adj).
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = m / np.arange(1, m + 1) * p[order]
    q = np.minimum(1.0, np.minimum.accumulate(q[::-1])[::-1])
    p_adj = np.empty(m)
    p_adj[order] = q
    return p_adj <= alpha, p_adj


def romano_wolf(returns_matrix, benchmark=0.0, n_boot=1000, alpha=0.05,
                studentized=True, block_size=None, seed=None):
    """Romano-Wolf stepdown FWER control via circular block bootstrap.

    Tests H0_i: E[r_i - b] <= 0 jointly for all N strategy columns while
    controlling the family-wise error rate. Because whole rows are
    resampled, the cross-sectional dependence between strategies is
    preserved — this is what makes Romano-Wolf less conservative than
    Bonferroni/Holm on correlated strategy panels.

    Algorithm (Romano & Wolf 2005; adjusted p-values per Romano & Wolf 2016):
      1. X = returns - benchmark, shape (T, N); test statistics
         t_i = sqrt(T) * mean_i / sd_i (studentized) or sqrt(T) * mean_i.
         Columns with zero variance get t = -inf and are never rejected.
      2. Circular block bootstrap (Politis & Romano 1992) with
         block_size = ceil(T ** (1/3)) by default — increase for strongly
         autocorrelated returns. Each replication resamples whole rows and
         computes the centered statistic t*_b,i = sqrt(T)*(mu*_b,i - mu_i)/sd*_b,i.
      3. Order statistics descending; for rank k the null envelope is
         M_k(b) = max_{j >= k} t*_b,(j) (reverse cumulative max), and
         p_(k) = (1 + #{b : M_k(b) >= t_(k)}) / (n_boot + 1).
      4. Enforce monotonicity p_adj_(k) = max(p_adj_(k-1), p_(k)); unsort.

    Deterministic for a fixed seed. Memory is O(n_boot * N); the
    (n_boot, T, N) cube is never materialized.

    Returns (reject_mask, adj_pvalues, t_stats).
    """
    X = np.asarray(returns_matrix, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    b = np.asarray(benchmark, dtype=float)
    X = X - (b[:, None] if b.ndim == 1 else b)
    T, N = X.shape
    if block_size is None:
        block_size = int(np.ceil(T ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    if studentized:
        t = np.where(sd > 0, np.sqrt(T) * np.divide(
            mu, sd, out=np.zeros_like(mu), where=sd > 0), -np.inf)
    else:
        t = np.sqrt(T) * mu

    n_blocks = int(np.ceil(T / block_size))
    boot_t = np.empty((n_boot, N))
    for bi in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:T] % T
        Xb = X[idx]
        mu_b = Xb.mean(axis=0)
        if studentized:
            sd_b = Xb.std(axis=0, ddof=1)
            boot_t[bi] = np.sqrt(T) * (mu_b - mu) / (sd_b + 1e-12)
        else:
            boot_t[bi] = np.sqrt(T) * (mu_b - mu)

    order = np.argsort(-t)  # descending; -inf columns sort last
    boot_sorted = boot_t[:, order]
    # M_k(b) = max over ranks >= k  ->  reverse cumulative max along ranks
    env = np.maximum.accumulate(boot_sorted[:, ::-1], axis=1)[:, ::-1]
    t_sorted = t[order]
    p_sorted = (1.0 + (env >= t_sorted[None, :]).sum(axis=0)) / (n_boot + 1.0)
    p_sorted = np.maximum.accumulate(p_sorted)
    adj_pvalues = np.empty(N)
    adj_pvalues[order] = p_sorted
    adj_pvalues = np.where(np.isneginf(t), 1.0, adj_pvalues)
    return adj_pvalues <= alpha, adj_pvalues, t

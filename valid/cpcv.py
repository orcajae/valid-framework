"""Combinatorial Purged Cross-Validation (CPCV).

Lopez de Prado (2018), Arian et al. (2024).
"""
import numpy as np
from itertools import combinations


def make_groups(n_samples, n_groups):
    """Assign samples to contiguous groups."""
    gs = n_samples // n_groups
    gids = np.zeros(n_samples, dtype=int)
    for g in range(n_groups):
        s = g * gs
        e = (g + 1) * gs if g < n_groups - 1 else n_samples
        gids[s:e] = g
    return gids


def cpcv_split(group_ids, test_groups, purge_bars, embargo_bars=0):
    """Generate (train_idx, test_idx) with purge and embargo."""
    n = len(group_ids)
    test_mask = np.isin(group_ids, test_groups)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    purge_mask = np.zeros(n, dtype=bool)
    for tg in sorted(test_groups):
        tg_idx = np.where(group_ids == tg)[0]
        s, e = tg_idx[0], tg_idx[-1]
        purge_mask[max(0, s - purge_bars):s] = True
        purge_mask[e + 1:min(n, e + 1 + purge_bars)] = True
        if embargo_bars > 0:
            purge_mask[max(0, s - purge_bars - embargo_bars):max(0, s - purge_bars)] = True
            purge_mask[min(n, e + 1 + purge_bars):min(n, e + 1 + purge_bars + embargo_bars)] = True

    train_idx = train_idx[~purge_mask[train_idx]]
    return train_idx, test_idx


def cpcv_paths(n_groups, k):
    """Generate all C(n_groups, k) test group combinations."""
    return list(combinations(range(n_groups), k))


def run_cpcv(X, y, model_fn, n_groups=6, k=2, purge_bars=20, embargo_bars=0,
             metric_fn=None):
    """Run full CPCV evaluation.

    Args:
        X: feature array (n_samples, n_features)
        y: label array (n_samples,)
        model_fn: callable() -> fitted model with .predict/.predict_proba
        n_groups: number of contiguous groups
        k: number of test groups per path
        purge_bars: purge window size
        embargo_bars: embargo window size
        metric_fn: callable(y_true, y_pred_proba) -> float

    Returns:
        dict with 'is_scores', 'oos_scores' arrays (n_paths,)
    """
    from sklearn.metrics import roc_auc_score
    if metric_fn is None:
        metric_fn = lambda yt, yp: roc_auc_score(yt, yp)

    gids = make_groups(len(X), n_groups)
    paths = cpcv_paths(n_groups, k)
    is_scores = []
    oos_scores = []

    for test_groups in paths:
        train_idx, test_idx = cpcv_split(gids, test_groups, purge_bars, embargo_bars)
        if len(test_idx) < 10 or len(train_idx) < 30:
            is_scores.append(np.nan)
            oos_scores.append(np.nan)
            continue
        try:
            model = model_fn()
            model.fit(X[train_idx], y[train_idx])
            prob_is = model.predict_proba(X[train_idx])[:, 1]
            prob_oos = model.predict_proba(X[test_idx])[:, 1]
            is_scores.append(metric_fn(y[train_idx], prob_is))
            oos_scores.append(metric_fn(y[test_idx], prob_oos))
        except Exception:
            is_scores.append(np.nan)
            oos_scores.append(np.nan)

    return {
        "is_scores": np.array(is_scores),
        "oos_scores": np.array(oos_scores),
    }

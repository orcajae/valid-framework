"""Multiple-testing stage: how many grid variants survive selection-aware tests?

Consumes results/variant_returns.npz (from run_variant_grid.py) and applies,
per variant: raw Sharpe p-value (Lo 2002), Bonferroni, Holm, Benjamini-
Hochberg, Romano-Wolf stepdown (block bootstrap), Deflated Sharpe Ratio
(Bailey & Lopez de Prado 2014), and the Harvey-Liu-Zhu t>3 rule of thumb.

H0 per variant: expected net return <= 0 (absolute, after 18bp costs).
Buy-and-hold context SRs live in variant_grid.csv.

This stage is a framework extension beyond the SSRN paper — its outputs are
live-computed for the current data tier, never imported from the paper.

Usage:
  python experiments/run_multiple_testing.py [--smoke]
Output:
  results/multiple_testing.csv + printed method-comparison table
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

try:
    from experiments import config
except ImportError:
    import config

from valid.metrics import deflated_sharpe_ratio
from valid.multiple_testing import (
    sharpe_pvalues, bonferroni, holm, benjamini_hochberg, romano_wolf,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_cfg = config.get_run_config(smoke=args.smoke)

    npz = np.load(config.RESULTS_DIR / "variant_returns.npz", allow_pickle=True)
    R, ids = npz["returns"], npz["ids"]
    T, N = R.shape

    pvals, t_stats = sharpe_pvalues(R)
    rej_bonf, p_bonf = bonferroni(pvals, args.alpha)
    rej_holm, p_holm = holm(pvals, args.alpha)
    rej_bh, p_bh = benjamini_hochberg(pvals, args.alpha)
    rej_rw, p_rw, _ = romano_wolf(R, n_boot=run_cfg.n_boot, alpha=args.alpha,
                                  seed=config.RANDOM_SEED)

    ann_sr = np.array([np.sqrt(252) * r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
                       for r in R.T])
    dsr = np.array([deflated_sharpe_ratio(
        r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0,
        N, T, stats.skew(r), stats.kurtosis(r, fisher=False))[0] for r in R.T])

    out = pd.DataFrame({
        "strategy_id": ids, "sr_annual": ann_sr, "t_stat": t_stats,
        "p_raw": pvals, "p_bonferroni": p_bonf, "p_holm": p_holm,
        "p_bh": p_bh, "p_romano_wolf": p_rw, "dsr": dsr,
        "rej_raw": pvals <= args.alpha, "rej_bonferroni": rej_bonf,
        "rej_holm": rej_holm, "rej_bh": rej_bh, "rej_romano_wolf": rej_rw,
        "rej_harvey_t3": t_stats > 3.0, "rej_dsr95": dsr > 0.95,
    })
    out_path = config.RESULTS_DIR / "multiple_testing.csv"
    out.to_csv(out_path, index=False)

    print(f"Multiple testing over N={N} variants, T={T} bars "
          f"(alpha={args.alpha}, n_boot={run_cfg.n_boot})")
    print(f"{'method':18s} {'survivors':>9s}")
    for name, col in [("raw p<alpha", "rej_raw"), ("Bonferroni", "rej_bonferroni"),
                      ("Holm", "rej_holm"), ("Benjamini-Hochberg", "rej_bh"),
                      ("Romano-Wolf", "rej_romano_wolf"),
                      ("Harvey t>3", "rej_harvey_t3"), ("DSR>0.95", "rej_dsr95")]:
        print(f"{name:18s} {int(out[col].sum()):>6d}/{N}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

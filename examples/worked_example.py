"""Worked example: the VALID checklist catching an overfit strategy.

Two deliberately generic strategies on deterministic synthetic data:

- MinedRSI ("the shiny one"): 180 fast RSI mean-reversion configs, the best one
  picked by in-sample GROSS Sharpe on the first 70% of the sample — two
  textbook mistakes at once (selecting on IS performance AND ignoring costs
  at selection time). Evaluation is always net of costs on the holdout.
  Expected to die on the selection-aware items (V6 permutation, V9
  baselines, V4 PBO) and under Romano-Wolf / DSR.
- HonestSMA ("the boring one"): a single SMA 50/200 crossover with an honest
  9-config neighborhood, evaluated on the full sample.

Every checklist input is computed live with valid/ functions; the PASS/FAIL
table in the README is transcribed from this script's output, never asserted.
Strategies, parameters, and data are textbook-generic — no proprietary
content. Deterministic: fixed seeds throughout.

Usage:
  python examples/worked_example.py [--data synthetic|real] [--smoke]
                                    [--out results/worked_example]
Outputs (in --out):
  report_overfit.md / report_baseline.md   VALIDReport markdown
  summary.csv                              scoreboard + key metrics
  fig_is_oos_scatter.png                   IS vs OOS SR of the mined grid
  fig_equity_curves.png                    net equity, mined vs honest vs B&H
  fig_survivors.png                        multiple-testing survivor counts
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments import config
from experiments.data_sources import load_ohlcv
from experiments.run_variant_grid import rsi, sma_cross_signal

from valid.checklist import VALIDChecker
from valid.costs import apply_costs
from valid.cpcv import make_groups, cpcv_split, cpcv_paths
from valid.metrics import annualized_sharpe, compute_pbo, var_sr_is, deflated_sharpe_ratio
from valid.multiple_testing import sharpe_pvalues, bonferroni, holm, benjamini_hochberg, romano_wolf

SEED = 42
IS_FRACTION = 0.70


def mined_rsi_grid():
    """360 fast RSI configs: 6 lookbacks x 5 entries x 6 exits x 2 directions
    (mean-reversion and its momentum mirror — miners try both).

    Deliberately restricted to SHORT lookbacks and extreme bands: fast rules
    cannot ride the data's slow regime trend, so any in-sample winner among
    them is fitting noise — which is the point."""
    return [(lb, en, ex, d)
            for lb in [2, 3, 4, 5, 7, 10]
            for en in [10, 15, 20, 25, 30]
            for ex in [70, 75, 80, 85, 90, 95]
            for d in ["mr", "mom"]]


def rsi_longshort_signal(close, lookback, entry, exit_, direction="mr"):
    """Long-short RSI rule: mean-reversion ("mr": +1 below entry, -1 above
    exit) or its momentum mirror ("mom": signs flipped). Long-short doubles
    the exposure to noise, which is exactly what makes this family a fertile
    mining ground."""
    r = rsi(close, lookback)
    sig = pd.Series(np.nan, index=close.index)
    sig[r < entry] = 1.0
    sig[r > exit_] = -1.0
    sig = sig.ffill().fillna(0)
    return sig if direction == "mr" else -sig


def honest_sma_grid():
    """The honest strategy's 9-config neighborhood."""
    return [(f, s) for f in [20, 50, 100] for s in [100, 200, 300]]


def net_returns_panel(close, ret, signals, cost_bp):
    """(T, n_configs) net-return matrix + per-config trade counts."""
    cols, trades = [], []
    for sig in signals:
        net, n_trades, _ = apply_costs(ret, sig, cost_bp)
        cols.append(net.values)
        trades.append(n_trades)
    return np.column_stack(cols), np.array(trades)


def cpcv_sr_matrices(panel, n_groups=6, k=2, purge=20):
    """(n_configs, n_paths) IS/OOS annualized-SR matrices over CPCV paths."""
    T, n_cfg = panel.shape
    gids = make_groups(T, n_groups)
    paths = cpcv_paths(n_groups, k)
    is_m = np.full((n_cfg, len(paths)), np.nan)
    oos_m = np.full((n_cfg, len(paths)), np.nan)
    for fi, tg in enumerate(paths):
        tr, te = cpcv_split(gids, tg, purge_bars=purge)
        for ci in range(n_cfg):
            is_m[ci, fi] = annualized_sharpe(panel[tr, ci])
            oos_m[ci, fi] = annualized_sharpe(panel[te, ci])
    return is_m, oos_m


def alignment_permutation_p(positions, returns, n_perm, seed=SEED):
    """Circular-shift permutation test: does the signal's ALIGNMENT with
    returns matter? Preserves the position series' autocorrelation."""
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(returns, dtype=float)
    obs = annualized_sharpe(pos * ret)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shift = rng.integers(1, len(pos))
        if annualized_sharpe(np.roll(pos, shift) * ret) >= obs:
            count += 1
    return (1 + count) / (n_perm + 1)


def baselines_net_sr(close, ret, eval_slice, cost_bp):
    """Paper-style simple baselines (buy & hold, RSI(14)>50, MACD cross).
    The SMA family is excluded because HonestSMA *is* an SMA rule —
    comparing a strategy against its own near-clone is not a baseline test."""
    out = {}
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    sigs = {
        "buy_hold": pd.Series(1.0, index=close.index),
        "rsi14_gt50": (rsi(close, 14) > 50).astype(float),
        "macd_cross": (ema12 > ema26).astype(float),
    }
    for name, sig in sigs.items():
        net, _, _ = apply_costs(ret, sig, cost_bp)
        out[name] = annualized_sharpe(net.values[eval_slice])
    return out


def evaluate_strategy(name, close, ret, regime, panel, sel_idx,
                      eval_slice, positions_eval, n_perm, cost_bp,
                      selection_panel=None):
    """Compute all 12 checklist inputs for one strategy; return (report, info).

    selection_panel: the (T, n_configs) return matrix matching the SELECTION
    criterion the researcher actually used — CPCV/PBO and Var(SR_IS) must
    diagnose the selection procedure itself (for MinedRSI that is gross
    returns, since it selected on gross IS Sharpe). Defaults to `panel`.
    """
    is_m, oos_m = cpcv_sr_matrices(
        panel if selection_panel is None else selection_panel)
    pbo, _, _ = compute_pbo(is_m, oos_m)
    v5 = var_sr_is(is_m)
    perm_p = alignment_permutation_p(positions_eval, ret.values[eval_slice], n_perm)

    net_eval = panel[eval_slice, sel_idx]
    gross = ret.values[eval_slice] * positions_eval
    gross_sr = annualized_sharpe(gross)
    net_sr = annualized_sharpe(net_eval)

    sr_at_costs = {}
    sel_signal = info_signals[name]
    for c in config.COST_LEVELS:
        net_c, _, _ = apply_costs(ret, sel_signal, c)
        sr_at_costs[c] = round(annualized_sharpe(net_c.values[eval_slice]), 3)

    years = max(len(net_eval) / 252, 1e-9)
    n_trades_eval = int((sel_signal.shift(1).fillna(0).diff().abs() > 0.5)
                        .values[eval_slice].sum())
    baselines = baselines_net_sr(close, ret, eval_slice, cost_bp)
    bear_mask = (regime == 1)[eval_slice]
    bear_sr = annualized_sharpe(net_eval[bear_mask]) if bear_mask.sum() > 50 else np.nan

    checker = VALIDChecker()
    report = checker.run_all(
        y_pred_unbal=positions_eval, y_pred_bal=positions_eval,
        has_temporal_split=True,
        pbo_value=round(float(pbo), 3), n_configs=panel.shape[1],
        var_sr_is=round(float(v5), 4),
        has_permutation_test=True, permutation_p=round(float(perm_p), 4),
        gross_sr=round(float(gross_sr), 3), net_sr=round(float(net_sr), 3),
        cost_bp=cost_bp, sr_at_costs=sr_at_costs,
        ml_sr=round(float(net_sr), 3), baseline_srs={k: round(v, 3) for k, v in baselines.items()},
        has_bear_market_eval=True,
        trades_per_year=round(n_trades_eval / years, 1),
        gross_alpha=round(float(gross_sr), 3),
        cost_drag=round(float(gross_sr - net_sr), 3),
        code_available=True,
    )
    info = {"strategy": name, "score": report.score, "total": report.total,
            "pbo": round(float(pbo), 3), "var_sr_is": round(float(v5), 4),
            "perm_p": round(float(perm_p), 4), "gross_sr": round(float(gross_sr), 3),
            "net_sr": round(float(net_sr), 3), "bear_sr": round(float(bear_sr), 3),
            "trades_yr": round(n_trades_eval / years, 1)}
    return report, info, (is_m, oos_m)


info_signals = {}  # name -> selected-config full-sample signal (for cost curves)


def main(data="synthetic", smoke=False, out_dir=None):
    run_cfg = config.get_run_config(smoke=smoke)
    n_perm = 50 if smoke else 200
    n_boot = run_cfg.n_boot
    out = Path(out_dir or (config.RESULTS_DIR / "worked_example"))
    out.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(data, "BTC", n_bars=run_cfg.n_bars)
    close = df["close"]
    ret = close.pct_change().fillna(0)
    T = len(close)
    regime = df.attrs.get("regime")
    if regime is None:  # real data: 2022 as the bear window
        regime = np.where((close.index >= "2022-01-01") & (close.index < "2023-01-01"), 1, 0)
    split = int(T * IS_FRACTION)
    holdout = slice(split, T)
    full = slice(0, T)

    # --- Strategy A: MinedRSI ------------------------------------------------
    grid_a = mined_rsi_grid()
    signals_a = [rsi_longshort_signal(close, lb, en, ex, d)
                 for lb, en, ex, d in grid_a]
    panel_a, trades_a = net_returns_panel(close, ret, signals_a, config.COST_RETAIL_BP)
    # Selection on IS *gross* SR — the miner's classic mistake (costs ignored
    # at selection time; all reported evaluation below is net of costs).
    gross_panel_a = np.column_stack(
        [(ret * sig.shift(1).fillna(0)).values for sig in signals_a])
    is_sr = np.array([annualized_sharpe(gross_panel_a[:split, i])
                      for i in range(gross_panel_a.shape[1])])
    sel_a = int(np.argmax(is_sr))
    lb, en, ex, d = grid_a[sel_a]
    print(f"MinedRSI: selected config rsi({lb},{en},{ex},{d}) by IS gross SR="
          f"{is_sr[sel_a]:.2f} over {len(grid_a)} configs; evaluating net on holdout")
    info_signals["MinedRSI"] = signals_a[sel_a]
    pos_a = signals_a[sel_a].shift(1).fillna(0).values[holdout]
    report_a, info_a, (is_m_a, oos_m_a) = evaluate_strategy(
        "MinedRSI", close, ret, regime, panel_a, sel_a,
        holdout, pos_a, n_perm, config.COST_RETAIL_BP,
        selection_panel=gross_panel_a)

    # --- Strategy B: HonestSMA ----------------------------------------------
    grid_b = honest_sma_grid()
    signals_b = [sma_cross_signal(close, f, s) for f, s in grid_b]
    panel_b, trades_b = net_returns_panel(close, ret, signals_b, config.COST_RETAIL_BP)
    sel_b = grid_b.index((50, 200))
    print(f"HonestSMA: pre-committed SMA(50,200), 9-config honest neighborhood; "
          f"evaluating on full sample")
    info_signals["HonestSMA"] = signals_b[sel_b]
    pos_b = signals_b[sel_b].shift(1).fillna(0).values
    report_b, info_b, _ = evaluate_strategy(
        "HonestSMA", close, ret, regime, panel_b, sel_b,
        full, pos_b, n_perm, config.COST_RETAIL_BP)

    # --- Multiple-testing epilogue on the mined grid -------------------------
    bh_net, _, _ = apply_costs(ret, pd.Series(1.0, index=close.index), config.COST_RETAIL_BP)
    pvals, t_stats = sharpe_pvalues(panel_a, benchmark=bh_net.values)
    rej_rw, _, _ = romano_wolf(panel_a, benchmark=bh_net.values,
                               n_boot=n_boot, seed=SEED)
    surv = {
        "raw p<0.05": int((pvals <= 0.05).sum()),
        "Bonferroni": int(bonferroni(pvals)[0].sum()),
        "Holm": int(holm(pvals)[0].sum()),
        "BH (FDR)": int(benjamini_hochberg(pvals)[0].sum()),
        "Romano-Wolf": int(rej_rw.sum()),
    }
    r_best = panel_a[:, sel_a]
    from scipy import stats as sps
    dsr_best, _, _ = deflated_sharpe_ratio(
        r_best.mean() / (r_best.std(ddof=1) + 1e-12), len(grid_a), T,
        sps.skew(r_best), sps.kurtosis(r_best, fisher=False))
    surv["DSR>0.95 (best)"] = int(dsr_best > 0.95)
    print(f"\nMultiple-testing epilogue over the {len(grid_a)}-config mined grid "
          f"(benchmark: buy-and-hold): {surv} | DSR(best)={dsr_best:.3f}")

    # --- Reports + summary ----------------------------------------------------
    print("\n===== MinedRSI ====="); report_a.print_summary()
    print("\n===== HonestSMA ====="); report_b.print_summary()
    report_a.to_markdown(out / "report_overfit.md")
    report_b.to_markdown(out / "report_baseline.md")
    summary = pd.DataFrame([info_a, info_b])
    summary["dsr_best_mined"] = [round(float(dsr_best), 4), np.nan]
    summary.to_csv(out / "summary.csv", index=False)

    # --- Figures --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_is, mean_oos = np.nanmean(is_m_a, axis=1), np.nanmean(oos_m_a, axis=1)
    ax.scatter(mean_is, mean_oos, s=12, alpha=0.5,
               label=f"{len(grid_a)} mined configs")
    ax.scatter(mean_is[sel_a], mean_oos[sel_a], s=90, marker="*", color="crimson",
               label="IS-selected config", zorder=5)
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("mean IS Sharpe (CPCV train folds)")
    ax.set_ylabel("mean OOS Sharpe (CPCV test folds)")
    ax.set_title("Data mining in one picture: IS rank does not survive OOS")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_is_oos_scatter.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, series in [("MinedRSI (IS-selected)", panel_a[:, sel_a]),
                          ("HonestSMA 50/200", panel_b[:, sel_b]),
                          ("Buy & Hold", bh_net.values)]:
        ax.plot(close.index, np.cumprod(1 + series), label=label, lw=1.2)
    ax.axvline(close.index[split], color="gray", ls=":", lw=1,
               label="IS / holdout boundary")
    ax.set_yscale("log"); ax.set_ylabel("net equity (18bp costs, log)")
    ax.set_title(f"Synthetic data (seed fixed) — net of {config.COST_RETAIL_BP}bp costs")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_equity_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(surv.keys())
    ax.bar(names, [surv[k] for k in names], color="steelblue", edgecolor="black")
    ax.set_ylabel(f"survivors / {len(grid_a)}")
    ax.set_title(f"Which of the {len(grid_a)} mined configs survive "
                 "selection-aware tests?")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(out / "fig_survivors.png", dpi=150); plt.close(fig)

    print(f"\nOutputs -> {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(data=args.data, smoke=args.smoke, out_dir=args.out)

"""Textbook strategy-variant grid for the multiple-testing stage.

Generates a panel of generic rule-based strategies (SMA crossover, RSI
mean-reversion, time-series momentum, N-bar breakout) per asset, long/flat,
T+1 execution, 18bp round-trip costs via valid.costs.apply_costs. This panel
feeds experiments/run_multiple_testing.py (Romano-Wolf / DSR / Bonferroni...).

These are deliberately standard, public-domain rules: the point is the
multiple-testing machinery, not the strategies. Not related to the SSRN
paper's 340-variant ML corpus (see results/reference/variants_340.csv).

Usage:
  python experiments/run_variant_grid.py [--data synthetic|real] [--smoke]
Outputs:
  results/variant_grid.csv     per-variant summary (gross/net SR, trades/yr)
  results/variant_returns.npz  returns (T,N), ids (N,), benchmark (T,) buy-hold
"""
import argparse

import numpy as np
import pandas as pd

try:
    from experiments import config
    from experiments.data_sources import load_ohlcv
except ImportError:
    import config
    from data_sources import load_ohlcv

from valid.costs import apply_costs
from valid.metrics import annualized_sharpe


def rsi(close, lookback):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(lookback).mean()
    loss = (-delta.clip(upper=0)).rolling(lookback).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sma_cross_signal(close, fast, slow):
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)


def rsi_meanrev_signal(close, lookback, entry, exit_):
    r = rsi(close, lookback)
    sig = pd.Series(np.nan, index=close.index)
    sig[r < entry] = 1.0
    sig[r > exit_] = 0.0
    return sig.ffill().fillna(0)


def momentum_signal(close, lookback):
    return (close.pct_change(lookback) > 0).astype(float)


def breakout_signal(close, n):
    return (close >= close.rolling(n).max()).astype(float).ffill().fillna(0)


def build_grid(small=False):
    """(family, id-suffix, signal_fn) tuples. `small` is the CI subset."""
    grid = []
    sma_f = [20, 50] if small else [10, 20, 50, 100]
    sma_s = [100, 200] if small else [100, 150, 200, 300]
    for f in sma_f:
        for s in sma_s:
            if f < s:
                grid.append(("sma_cross", f"sma_{f}_{s}",
                             lambda c, f=f, s=s: sma_cross_signal(c, f, s)))
    rsi_lb = [2, 14] if small else [2, 5, 14, 21]
    rsi_en = [30] if small else [20, 25, 30]
    rsi_ex = [60] if small else [50, 60, 70]
    for lb in rsi_lb:
        for en in rsi_en:
            for ex in rsi_ex:
                grid.append(("rsi_meanrev", f"rsi_{lb}_{en}_{ex}",
                             lambda c, lb=lb, en=en, ex=ex: rsi_meanrev_signal(c, lb, en, ex)))
    for lb in ([126] if small else [21, 63, 126, 252]):
        grid.append(("momentum", f"mom_{lb}",
                     lambda c, lb=lb: momentum_signal(c, lb)))
    for n in ([55] if small else [20, 55, 100]):
        grid.append(("breakout", f"brk_{n}",
                     lambda c, n=n: breakout_signal(c, n)))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_cfg = config.get_run_config(smoke=args.smoke)

    grid = build_grid(small=run_cfg.grid == "small" or args.smoke)
    assets = ["BTC", "ETH", "SOL"]

    frames = {a: load_ohlcv(args.data, a, n_bars=run_cfg.n_bars) for a in assets}
    common = frames[assets[0]].index
    for a in assets[1:]:
        common = common.intersection(frames[a].index)

    rows, cols, ids = [], [], []
    bench_parts = []
    for a in assets:
        close = frames[a].loc[common, "close"]
        ret = close.pct_change().fillna(0)
        bench_parts.append(ret)
        for family, suffix, fn in grid:
            sig = fn(close)
            net, n_trades, drag = apply_costs(ret, sig, config.COST_RETAIL_BP)
            gross = ret * sig.shift(1).fillna(0)
            years = max(len(ret) / 252, 1e-9)
            sid = f"{a}_{suffix}"
            rows.append({
                "strategy_id": sid, "asset": a, "family": family,
                "gross_sr": annualized_sharpe(gross.values),
                "net_sr_18bp": annualized_sharpe(net.values),
                "trades_yr": n_trades / years, "cost_drag_pct": drag,
                "T": len(net),
            })
            cols.append(net.values)
            ids.append(sid)

    # Benchmark: equal-weight buy-and-hold of the three assets.
    benchmark = np.mean(np.column_stack([b.values for b in bench_parts]), axis=1)

    grid_df = pd.DataFrame(rows)
    grid_path = config.RESULTS_DIR / "variant_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    np.savez(config.RESULTS_DIR / "variant_returns.npz",
             returns=np.column_stack(cols), ids=np.array(ids), benchmark=benchmark)
    print(f"Saved {len(ids)} variants ({args.data}) -> {grid_path} + variant_returns.npz")
    print(f"  net SR: median={grid_df.net_sr_18bp.median():.3f} "
          f"max={grid_df.net_sr_18bp.max():.3f} "
          f"negative={((grid_df.net_sr_18bp < 0).mean() * 100):.0f}%")


if __name__ == "__main__":
    main()

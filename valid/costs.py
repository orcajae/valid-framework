"""Transaction cost models."""
import numpy as np
import pandas as pd


def apply_costs(daily_returns, position, cost_rt_bp=18):
    """Apply round-trip transaction costs to strategy returns.

    Args:
        daily_returns: pd.Series of asset daily returns
        position: pd.Series of positions (0=flat, 1=long), T+1 applied
        cost_rt_bp: round-trip cost in basis points

    Returns:
        (net_returns, n_trades, cost_drag_pct)
    """
    cost = cost_rt_bp / 10000
    pos = position.shift(1).fillna(0)
    trades = (pos.diff().abs() > 0.5).astype(int)
    trades.iloc[0] = 0

    ret_gross = daily_returns * pos
    trade_costs = trades * cost
    ret_net = ret_gross - trade_costs

    n_trades = int(trades.sum())
    gross_total = (1 + ret_gross).prod() - 1
    cost_total = trade_costs.sum()
    drag = cost_total / max(gross_total, 0.001) * 100

    return ret_net, n_trades, drag

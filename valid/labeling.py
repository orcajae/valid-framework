"""Triple Barrier labeling + CUSUM event filter.

Lopez de Prado (2018).
"""
import numpy as np
import pandas as pd


def cusum_filter(prices, threshold):
    """CUSUM event detection on price series."""
    events = []
    s_pos, s_neg = 0.0, 0.0
    ret = prices.pct_change().dropna()
    for i in range(len(ret)):
        s_pos = max(0, s_pos + ret.iloc[i])
        s_neg = min(0, s_neg + ret.iloc[i])
        if s_pos > threshold:
            events.append(ret.index[i])
            s_pos = 0
        elif s_neg < -threshold:
            events.append(ret.index[i])
            s_neg = 0
    return pd.DatetimeIndex(events)


def triple_barrier_labels(df, event_idx, pt_mult, sl_mult, max_hold, vol_window):
    """Apply triple barrier labeling.

    Args:
        df: OHLCV DataFrame with 'close' column
        event_idx: DatetimeIndex of CUSUM events
        pt_mult: profit-taking barrier in vol multiples
        sl_mult: stop-loss barrier in vol multiples
        max_hold: max holding period in bars
        vol_window: EWMA volatility lookback

    Returns:
        pd.Series of labels {-1, 0, 1} indexed by event dates
    """
    c = df["close"]
    vol = c.pct_change().ewm(span=vol_window).std()
    labels = {}

    for t in event_idx:
        if t not in c.index:
            continue
        loc = c.index.get_loc(t)
        if loc + 1 >= len(c):
            continue
        entry = c.iloc[loc]
        v = vol.iloc[loc]
        if pd.isna(v) or v == 0:
            continue

        pt_barrier = entry * (1 + pt_mult * v)
        sl_barrier = entry * (1 - sl_mult * v)
        end_loc = min(loc + max_hold, len(c) - 1)

        label = 0
        for j in range(loc + 1, end_loc + 1):
            if c.iloc[j] >= pt_barrier:
                label = 1
                break
            elif c.iloc[j] <= sl_barrier:
                label = -1
                break
        labels[t] = label

    return pd.Series(labels, name="tb_label")

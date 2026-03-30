"""Feature engineering for crypto ML."""
import numpy as np
import pandas as pd


def compute_features(df):
    """Compute ~25 price-derived features from OHLCV DataFrame."""
    c = df["close"]
    v = df["volume"]
    ret1 = c.pct_change()
    feats = pd.DataFrame(index=df.index)

    for p in [1, 5, 10, 20, 60, 120]:
        feats[f"ret_{p}"] = c.pct_change(p)
    for p in [14, 30, 60]:
        feats[f"vol_{p}"] = ret1.rolling(p).std()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feats["macd_norm"] = (ema12 - ema26) / ema26.replace(0, np.nan)
    signal = (ema12 - ema26).ewm(span=9).mean()
    feats["macd_hist"] = (ema12 - ema26 - signal) / ema26.replace(0, np.nan)

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feats["bb_pos"] = (c - sma20) / (2 * std20).replace(0, np.nan)
    feats["bb_width"] = (4 * std20) / sma20.replace(0, np.nan)

    feats["vol_z_14"] = (v - v.rolling(14).mean()) / v.rolling(14).std().replace(0, np.nan)
    feats["range_pct"] = ((df["high"] - df["low"]) / c).rolling(14).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feats["atr_norm"] = tr.rolling(14).mean() / c

    return feats

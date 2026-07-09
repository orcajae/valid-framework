"""Tests for the synthetic OHLCV generator (experiments/data_sources.py)."""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.data_sources import make_synthetic_ohlcv, load_ohlcv


def test_deterministic_same_seed():
    a = make_synthetic_ohlcv(n_bars=500, seed=7)
    b = make_synthetic_ohlcv(n_bars=500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(a.attrs["regime"], b.attrs["regime"])


def test_different_seeds_differ():
    a = make_synthetic_ohlcv(n_bars=500, seed=7)
    b = make_synthetic_ohlcv(n_bars=500, seed=8)
    assert not np.allclose(a["close"].values, b["close"].values)


def test_no_nans_and_positive_prices():
    df = make_synthetic_ohlcv(n_bars=2200, seed=7)
    assert not df.isna().any().any()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()


def test_regime_attr_present_with_real_bear_share():
    df = make_synthetic_ohlcv(n_bars=2200, seed=7)
    regime = df.attrs["regime"]
    assert len(regime) == len(df)
    bear_share = (regime == 1).mean()
    assert 0.10 < bear_share < 0.60  # bear regime must actually occur


def test_load_ohlcv_asset_offsets():
    btc = load_ohlcv("synthetic", "BTC", n_bars=300)
    eth = load_ohlcv("synthetic", "ETH", n_bars=300)
    assert not np.allclose(btc["close"].values, eth["close"].values)

"""Unit tests for labeling."""
import numpy as np
import pandas as pd
from valid.labeling import cusum_filter, triple_barrier_labels


def test_cusum_filter():
    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.randn(1000)) + 100,
                       index=pd.date_range("2020-01-01", periods=1000, freq="h"))
    events = cusum_filter(prices, 0.05)
    assert len(events) > 0
    assert len(events) < len(prices)


def test_triple_barrier():
    np.random.seed(42)
    n = 500
    prices = np.cumsum(np.random.randn(n) * 0.01) + 100
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.random.lognormal(10, 1, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="h"))
    events = cusum_filter(df["close"], 0.03)
    if len(events) > 0:
        labels = triple_barrier_labels(df, events, 1.5, 2.0, 20, 60)
        assert set(labels.values).issubset({-1, 0, 1})

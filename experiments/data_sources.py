"""Data sources for the reproduction pipeline.

Two tiers (see REPRODUCE.md):
- "synthetic": deterministic regime-switching GBM. Offline- and CI-safe;
  verifies the pipeline mechanism, not the published real-data numbers.
- "real": public Binance OHLCV CSVs fetched by experiments/download_data.py.
"""
import numpy as np
import pandas as pd

try:  # runnable both as a package module and as a script from experiments/
    from experiments import config
except ImportError:
    import config

# Per-regime daily drift/vol; a persistent Markov chain switches between them.
# Parameters were chosen ONCE so that trend-following has a real (modest)
# edge and bear regimes genuinely hurt buy-and-hold — the two pedagogical
# outcomes the worked example needs to be visible. They are never tuned
# per-run; changing them changes all synthetic-tier outputs.
REGIMES = {
    0: {"name": "bull", "mu": 0.0040, "sigma": 0.026},
    1: {"name": "bear", "mu": -0.0055, "sigma": 0.048},
    2: {"name": "chop", "mu": 0.0000, "sigma": 0.018},
}
# Expected regime length: bull ~125 bars, bear ~100, chop ~33 — bear
# regimes must outlast a slow moving-average's response time, and the
# regime edge must be large enough to be statistically detectable at
# T~2200, for trend-following to have a real, testable edge.
TRANSITION = np.array([
    [0.992, 0.004, 0.004],
    [0.007, 0.990, 0.003],
    [0.015, 0.015, 0.970],
])
STUDENT_T_DF = 5  # fat-tailed innovations

ASSET_SEED_OFFSET = {"BTC": 0, "ETH": 1, "SOL": 2}


def make_synthetic_ohlcv(n_bars=2200, seed=7, start="2019-01-01", freq="D"):
    """Deterministic regime-switching GBM OHLCV frame.

    The generating regime path is exposed in df.attrs["regime"] (int array,
    see REGIMES) so bear-market evaluation (VALID item V10) can condition on
    the exact simulated regime rather than an eyeballed window.
    Same (n_bars, seed) => bit-identical output.
    """
    rng = np.random.default_rng(seed)
    regime = np.empty(n_bars, dtype=int)
    regime[0] = 0
    for i in range(1, n_bars):
        regime[i] = rng.choice(3, p=TRANSITION[regime[i - 1]])

    mu = np.array([REGIMES[r]["mu"] for r in regime])
    sigma = np.array([REGIMES[r]["sigma"] for r in regime])
    # Student-t innovations scaled to unit variance for interpretable sigma.
    t_scale = np.sqrt(STUDENT_T_DF / (STUDENT_T_DF - 2))
    innov = rng.standard_t(STUDENT_T_DF, size=n_bars) / t_scale
    log_ret = mu + sigma * innov

    close = 10000 * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[10000.0], close[:-1]])
    intrabar = np.abs(rng.normal(0, 0.5, size=(2, n_bars))) * sigma
    high = np.maximum(open_, close) * (1 + intrabar[0])
    low = np.minimum(open_, close) * (1 - intrabar[1])
    volume = rng.lognormal(20, 1, n_bars)

    idx = pd.date_range(start, periods=n_bars, freq=freq)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.attrs["regime"] = regime
    df.attrs["source"] = f"synthetic(seed={seed}, n_bars={n_bars})"
    return df


def real_data_path(asset, timeframe="1d"):
    return config.RAW_DIR / f"binance_{asset.lower()}usdt_ohlcv_{timeframe}.csv"


def load_ohlcv(source, asset, timeframe="1d", seed=7, n_bars=2200):
    """Load OHLCV for an asset from the requested source.

    source="synthetic": per-asset seed offset keeps the three assets distinct
    but individually deterministic. source="real": CSV written by
    experiments/download_data.py (raises with instructions if absent).
    """
    if source == "synthetic":
        return make_synthetic_ohlcv(
            n_bars=n_bars, seed=seed + ASSET_SEED_OFFSET.get(asset.upper(), 0)
        )
    if source == "real":
        path = real_data_path(asset, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python experiments/download_data.py` "
                f"(requires the [data] extra: pip install -e '.[data]')"
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        df.attrs["source"] = str(path)
        return df
    raise ValueError(f"unknown source {source!r} (expected 'synthetic' or 'real')")

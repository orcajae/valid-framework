# Data Directory

No data is committed to git (see .gitignore). Two ways to get data:

- **Synthetic (default, no download):** every pipeline stage defaults to a
  deterministic regime-switching generator — see
  `experiments/data_sources.py` (`make_synthetic_ohlcv`, fixed seeds,
  bit-reproducible). Nothing needs to exist in this directory.
- **Real (public Binance OHLCV):** run

  ```bash
  pip install -e ".[data]"          # installs ccxt
  python experiments/download_data.py
  ```

  which writes `data/raw/binance_{asset}usdt_ohlcv_1d.csv` for BTC/ETH/SOL
  plus a SHA-256 manifest (`data/raw/CHECKSUMS.sha256`) for provenance.
  Public endpoints only — no API key needed.

See REPRODUCE.md for how the two tiers relate to the published numbers.

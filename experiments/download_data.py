"""Download public Binance OHLCV via ccxt for the `--real` reproduction tier.

Usage:
  python experiments/download_data.py [--assets BTC ETH SOL] [--timeframe 1d]
                                      [--since 2019-01-01]

Writes data/raw/binance_{asset}usdt_ohlcv_{timeframe}.csv plus a SHA-256
checksum manifest (data/raw/CHECKSUMS.sha256) for provenance. Public
endpoints only — no API key required.
"""
import argparse
import hashlib
import time

import pandas as pd

try:
    from experiments import config
    from experiments.data_sources import real_data_path
except ImportError:
    import config
    from data_sources import real_data_path


def fetch_ohlcv(exchange, symbol, timeframe, since_ms):
    """Paginated OHLCV fetch (mirrors the starter-notebook loop)."""
    rows = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def write_checksums():
    manifest = config.RAW_DIR / "CHECKSUMS.sha256"
    lines = []
    for p in sorted(config.RAW_DIR.glob("*.csv")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {p.name}")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--since", default="2019-01-01")
    args = parser.parse_args()

    try:
        import ccxt
    except ImportError:
        raise SystemExit(
            "ccxt is required for real-data download: pip install -e '.[data]'"
        )

    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(args.since).timestamp() * 1000)

    for asset in args.assets:
        symbol = f"{asset.upper()}/USDT"
        out = real_data_path(asset, args.timeframe)
        print(f"Fetching {symbol} {args.timeframe} since {args.since} ...")
        df = fetch_ohlcv(exchange, symbol, args.timeframe, since_ms)
        df.to_csv(out)
        print(f"  {len(df)} bars -> {out}")

    manifest = write_checksums()
    print(f"Checksums -> {manifest}")


if __name__ == "__main__":
    main()

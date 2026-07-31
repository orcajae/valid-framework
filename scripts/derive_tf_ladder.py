#!/usr/bin/env python3
"""
Derive the trading-frequency ladder (camera-ready Table 5) from the 340-variant corpus.

Source : results/reference/variants_340.csv
Filter : real model variants only — COST_* (cost-sweep series) and BASE_*
         (rule-based baselines) rows carry no model prediction and are excluded,
         as are the cross-section (XS_*) and SHAP rule (SHAP_*) rows, which are
         not per-timeframe model variants.
Output : per-timeframe n / median / best / share of negative net SR at 18 bp.

No model is trained and no backtest is run here: the script only aggregates
values already present in the corpus file.
"""
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "reference", "variants_340.csv")
EXCLUDE_PREFIX = ("COST_", "BASE_", "MC_", "XS_", "SHAP_")
ORDER = ["15m", "1h", "4h", "1d"]
BENCH = 0.917


def main():
    rows = list(csv.DictReader(open(SRC)))
    print(f"source            : {os.path.relpath(SRC, ROOT)}")
    print(f"rows in corpus    : {len(rows)}")

    kept = [r for r in rows
            if not r["strategy_id"].startswith(EXCLUDE_PREFIX)
            and r["timeframe"] in ORDER]
    print(f"excluded prefixes : {', '.join(EXCLUDE_PREFIX)}")
    print(f"model variants    : {len(kept)}\n")

    print(f"{'TF':<5}{'n':>6}{'median':>10}{'best':>10}{'worst':>10}{'share<0':>10}")
    print("-" * 51)
    for tf in ORDER:
        v = sorted(float(r["net_sr_18bp"]) for r in kept if r["timeframe"] == tf)
        neg = 100.0 * sum(1 for x in v if x < 0) / len(v)
        print(f"{tf:<5}{len(v):>6}{st.median(v):>10.3f}{max(v):>10.3f}"
              f"{min(v):>10.3f}{neg:>9.0f}%")

    print("\nrounded to the two decimals printed in Table 5:")
    for tf in ORDER:
        v = sorted(float(r["net_sr_18bp"]) for r in kept if r["timeframe"] == tf)
        neg = 100.0 * sum(1 for x in v if x < 0) / len(v)
        print(f"  {tf:<4} n={len(v):<4} median={st.median(v):+.2f}  "
              f"best={max(v):+.2f}  share<0={neg:.0f}%")
    print(f"\nbenchmark net SR (DM 252/126+CB, 18 bp) = {BENCH}")


if __name__ == "__main__":
    main()

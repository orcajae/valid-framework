#!/usr/bin/env python3
"""
Recompute the corpus-level headline figures from the 340-variant corpus, and
report how they change when the cost-level entries are removed.

Source : results/reference/variants_340.csv
Prints : (a) full corpus (the printed figures: 52% negative, 15 above benchmark)
         (b) corpus without the 18 COST_* cost-level entries
         (c) the identifiers of every variant exceeding the benchmark
         (d) corpus without every entry traced to a synthetic signal proxy

The benchmark threshold is the rounded net Sharpe of the DM 252/126+CB
benchmark (0.917) as printed in the paper; the full-precision comparison is also
reported because two variants sit between the rounded and full-precision
thresholds (documented in REPRODUCE.md).

Provenance of the excluded rows (see poster/RUN_ATTRIBUTION_AUDIT.md):
  COST_benchmark_*   rule-based DM+CB benchmark at six cost levels   (real)
  COST_ML_1h_*       signal proxy `ml_sig`      at six cost levels   (synthetic)
  COST_ML_15m_*      signal proxy `ml_15m_sig`  at six cost levels   (synthetic)
  BASE_1h ML balanced  the same `ml_sig` proxy at 18 bp             (synthetic)
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "reference", "variants_340.csv")
BENCH_ROUNDED = 0.917
BENCH_FULL = 0.9173201080439872   # results/reference/traditional_baselines.csv
SYNTHETIC_MODELS = ("ML_1h", "ML_15m")
SYNTHETIC_IDS = ("BASE_1h ML balanced",)


def stats(rows, label):
    sr = [float(r["net_sr_18bp"]) for r in rows]
    n = len(sr)
    neg = sum(1 for x in sr if x < 0)
    hi_r = sum(1 for x in sr if x > BENCH_ROUNDED)
    hi_f = sum(1 for x in sr if x > BENCH_FULL)
    print(f"{label}")
    print(f"  n                      : {n}")
    print(f"  negative net SR        : {neg}/{n} = {100.0 * neg / n:.1f}%")
    print(f"  > {BENCH_ROUNDED} (rounded)      : {hi_r}/{n} = {100.0 * hi_r / n:.1f}%")
    print(f"  > {BENCH_FULL:.6f} (full) : {hi_f}/{n} = {100.0 * hi_f / n:.1f}%")
    return n, neg, hi_r


def main():
    rows = list(csv.DictReader(open(SRC)))
    print(f"source : {os.path.relpath(SRC, ROOT)}\n")

    print("(a) full corpus — the figures printed in the paper")
    stats(rows, "    all variants")

    cost = [r for r in rows if r["strategy_id"].startswith("COST_")]
    rest = [r for r in rows if not r["strategy_id"].startswith("COST_")]
    print(f"\n(b) excluding the {len(cost)} COST_* cost-level entries")
    stats(rest, "    remaining variants")

    print("\n(c) variants exceeding the benchmark (rounded threshold)")
    top = sorted((r for r in rows if float(r["net_sr_18bp"]) > BENCH_ROUNDED),
                 key=lambda r: -float(r["net_sr_18bp"]))
    for r in top:
        tag = ""
        if r["strategy_id"].startswith(("COST_", "BASE_")):
            tag = "   <-- cost-level / baseline entry"
        print(f"    {r['strategy_id']:<28}{float(r['net_sr_18bp']):>8.4f}"
              f"  {r['timeframe']:<5}{tag}")
    print(f"    total: {len(top)}")

    syn = [r for r in rows
           if r["model"] in SYNTHETIC_MODELS or r["strategy_id"] in SYNTHETIC_IDS]
    non_syn = [r for r in rows if r not in syn]
    print(f"\n(d) excluding all {len(syn)} entries traced to a synthetic signal proxy")
    stats(non_syn, "    remaining variants")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Print the multiple-testing figures of camera-ready Table 8 / Section 4.5
from the stored experiment output.

Source : ~/jwquant/paper/kdd-mlf/results/multiple_testing.json
         (override with MT_JSON=<path>)

The JSON is the output of run_multiple_testing_v2.py, which reads the
340-variant corpus; this script only reads and formats it.
"""
import json
import os

DEFAULT = os.path.expanduser(
    "~/jwquant/paper/kdd-mlf/results/multiple_testing.json")
SRC = os.environ.get("MT_JSON", DEFAULT)


def main():
    d = json.load(open(SRC))
    print(f"source            : {SRC}")
    print(f"total variants    : {d['total_variants']}")
    print(f"benchmark SR      : {d['benchmark_sr']}")
    print(f"best observed SR  : {d['best_observed_sr']}")
    print(f"E[max SR] null    : {d['E_max_sr_null']}\n")

    print(f"{'method':<20}{'n survive':>10}{'pct':>8}")
    print("-" * 38)
    for k, v in d["methods"].items():
        print(f"{k:<20}{v['n']:>10}{v['pct']:>7}%")

    surv = d["bonferroni_survivors"]
    print(f"\nBonferroni survivors: {len(surv)}")
    print(f"{'strategy_id':<28}{'SR':>7}{'t':>8}{'PBO':>6}{'AUC':>7}  TF")
    print("-" * 62)
    for s in surv:
        print(f"{s['strategy_id']:<28}{s['sr']:>7.2f}{s['t_stat']:>8.2f}"
              f"{s['pbo']:>6.1f}{s['auc']:>7.3f}  {s['timeframe']}")
    ts = [s["t_stat"] for s in surv]
    aucs = [s["auc"] for s in surv]
    pbos = {s["pbo"] for s in surv}
    print(f"\nt range   : {min(ts):.2f} – {max(ts):.2f}")
    print(f"AUC range : {min(aucs):.2f} – {max(aucs):.2f}")
    print(f"PBO values: {sorted(pbos)}")
    print(f"key finding: {d['key_finding']}")


if __name__ == "__main__":
    main()

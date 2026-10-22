"""Robustness study of the E14 findings (CHANGELOG E15): merge parts, write the pre-registered tables and the mechanical decision file.

usage: python3 -I experiments/inbox7/summarize.py 1|2      (from the repository root)
  1: results/inbox7/stage1_{rows.csv, arrays.npz, rates.csv, mean_pbo.csv, tables.json}, decision.json (D1)
  2: results/inbox7/stage2_{...}, decision.json (D2 added; D1 kept)
Every verdict follows section 3 of the pre-registration literally; nothing is chosen here.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

STAGE = int(sys.argv[1])
D = "results/inbox7"
P = os.path.join(D, "parts")
BOOT_B, BOOT_SEED = 10000, 0
THR = (0.20, 0.50)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    a = z / d * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, c - a), min(1.0, c + a)


def merge():
    parts = sorted(glob.glob(os.path.join(P, f"s{STAGE}_*.csv")))
    df = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in parts], ignore_index=True)
    df["arm_id"] = np.where(df["arm"] == "A", "A-" + df["asset"], df["arm"])
    df = df.sort_values(["arm_id", "seed"]).reset_index(drop=True)
    df.to_csv(os.path.join(D, f"stage{STAGE}_rows.csv"), index=False)
    arrs = {}
    for f in parts:
        z = np.load(f[:-4] + ".npz")
        arrs.update({k: z[k] for k in z.files})
    np.savez_compressed(os.path.join(D, f"stage{STAGE}_arrays.npz"), **arrs)
    return df, parts


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def stats_stage1(g):
    out = {}
    for s in ("S1", "S2", "S3", "S4"):
        for t in THR:
            out[f"{s}_PBO<{t:.2f}"] = g[f"pbo_{s}"].astype(float) < t   # NaN -> False
    for h in ("u", "p"):
        out[f"AUC>0.55_{h}"] = g[f"auc_{h}"] > 0.55
        out[f"NetSR>0_{h}"] = g[f"net_sr_pe_{h}"] > 0
        out[f"G_{h}"] = as_bool(g[f"G_{h}"])
    return out


def stats_stage2(g):
    out = {}
    hold = {"T1": "u", "T2": "p", "T3": "p", "T4": "p"}
    for h in ("u", "p"):
        out[f"AUC>0.55_{h}"] = g[f"auc_{h}"] > 0.55
        out[f"NetSR>0_{h}"] = g[f"net_sr_pe_{h}"] > 0
        out[f"G_{h}"] = as_bool(g[f"G_{h}"])
        out[f"G'_{h}"] = as_bool(g[f"Gp_{h}"])
    for t, h in hold.items():
        out[t] = as_bool(g[f"{t}_pass"])
    # conjunctions of AUC, each permutation variant and each net-Sharpe rule on the same holdout
    for t, h in hold.items():
        for net in ("NetSR>0", "G'"):
            base = out[f"{net}_{h}"]
            out[f"AUC>0.55_{h} & {t}"] = out[f"AUC>0.55_{h}"] & out[t]
            out[f"AUC>0.55_{h} & {t} & {net}_{h}"] = out[f"AUC>0.55_{h}"] & out[t] & base
            out[f"{t} & {net}_{h}"] = out[t] & base
    return out


def rate_table(df, fn):
    rows = []
    for arm, g in df.groupby("arm_id"):
        g = g[g["valid"] == 1]
        n = len(g)
        for name, m in fn(g).items():
            k = int(m.sum())
            lo, hi = wilson(k, n)
            rows.append({"arm": arm, "statistic": name, "k": k, "n": n, "rate": k / n, "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.RandomState(BOOT_SEED)
    bs = x[rng.randint(0, len(x), size=(BOOT_B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def get(rates, arm, stat):
    r = rates[(rates.arm == arm) & (rates.statistic == stat)].iloc[0]
    return int(r.k), int(r.n)


def compare(rates, stat):
    """size = B0, power = B2, one-sided Fisher exact test of power > size."""
    ks, ns = get(rates, "B0", stat)
    kp, np_ = get(rates, "B2", stat)
    _, p = fisher_exact([[kp, np_ - kp], [ks, ns - ks]], alternative="greater")
    return {"statistic": stat, "size_k": ks, "size_n": ns, "size": ks / ns, "power_k": kp, "power_n": np_,
            "power": kp / np_, "diff_pp": 100 * (kp / np_ - ks / ns), "fisher_p_one_sided": float(p)}


def d1(rates):
    variants = {"V1": "S1", "V2": "S2", "V3": "S3", "V4": "S4"}
    cells = []
    for v, s in variants.items():
        for t in THR:
            c = compare(rates, f"{s}_PBO<{t:.2f}")
            if c["diff_pp"] >= 20 and c["fisher_p_one_sided"] < 0.05:
                c["verdict"] = "has power"
            elif c["diff_pp"] < 10 and c["fisher_p_one_sided"] >= 0.05:
                c["verdict"] = "no power"
            else:
                c["verdict"] = "inconclusive"
            c["variant"], c["threshold"] = v, t
            cells.append(c)
    by = lambda v: [c for c in cells if c["variant"] == v]  # noqa: E731
    with_power = [c for c in cells if c["variant"] in ("V2", "V3", "V4") and c["verdict"] == "has power"]
    no_both = {v: all(c["verdict"] == "no power" for c in by(v)) for v in variants}
    if with_power:
        outcome = "A variant with power exists"
        for c in with_power:
            c["size_in_A_and_C"] = {a: dict(zip(("k", "n"), get(rates, a, c["statistic"]))) for a in ("A-BTC", "C")}
    elif no_both["V1"] and sum(no_both[v] for v in ("V2", "V3", "V4")) >= 2:
        outcome = "Not a test for an edge"
    else:
        outcome = "Condition-dependent"
    return {"cells": cells, "no_power_at_both_thresholds": no_both,
            "variants_with_power": [f'{c["variant"]}@{c["threshold"]}' for c in with_power], "outcome": outcome}


def d2(rates):
    cells = []
    for t in ("T2", "T3", "T4"):
        per = {a: dict(zip(("k", "n"), get(rates, f"A-{a}", t))) for a in ("BTC", "ETH", "SOL")}
        K, N = sum(v["k"] for v in per.values()), sum(v["n"] for v in per.values())
        lo, hi = wilson(K, N)
        each_ok = all(v["k"] / v["n"] <= 0.075 for v in per.values())
        kp, np_ = get(rates, "B2", t)
        cells.append({"statistic": t, "A_per_asset": {a: {**v, "rate": v["k"] / v["n"]} for a, v in per.items()},
                      "each_le_7.5pct": each_ok, "pooled_k": K, "pooled_n": N, "pooled_wilson_hi": hi,
                      "pooled_hi_le_9pct": bool(hi <= 0.09), "size_controlled": bool(each_ok and hi <= 0.09),
                      "B2_k": kp, "B2_n": np_, "B2_rate": kp / np_, "power_kept": bool(kp / np_ >= 0.50)})
    if any(c["size_controlled"] and c["power_kept"] for c in cells):
        outcome = "design found"
    elif any(c["size_controlled"] for c in cells):
        outcome = "partial"
    else:
        outcome = "none"
    return {"cells": cells, "outcome": outcome}


df, parts = merge()
rates = rate_table(df, stats_stage1 if STAGE == 1 else stats_stage2)
rates.to_csv(os.path.join(D, f"stage{STAGE}_rates.csv"), index=False)
tables = {"parts": [os.path.basename(p) for p in parts],
          "n_valid": df.groupby("arm_id")["valid"].sum().astype(int).to_dict(),
          "n_rows": df.groupby("arm_id").size().astype(int).to_dict()}
if STAGE == 1:
    mp = []
    for arm, g in df[df.valid == 1].groupby("arm_id"):
        for s in ("S1", "S2", "S3", "S4"):
            m, lo, hi, n = mean_ci(g[f"pbo_{s}"])
            mp.append({"arm": arm, "statistic": s, "mean_pbo": m, "boot_lo": lo, "boot_hi": hi, "n": n})
    pd.DataFrame(mp).to_csv(os.path.join(D, "stage1_mean_pbo.csv"), index=False)
    v = df[df.valid == 1]
    tables["mean_holdout_auc"] = {a: {h: float(v[v.arm_id == a][f"auc_{h}"].mean()) for h in ("u", "p")}
                                  for a in ("B0", "B1", "B2", "B0p")}
    tables["mean_events"] = v.groupby("arm_id")["events"].mean().round(1).to_dict()
    tables["pbo_nan_count"] = {s: int(v[f"pbo_{s}"].isna().sum()) for s in ("S1", "S2", "S3", "S4")}
    tables["paths"] = {s: v[f"paths_{s}"].value_counts().to_dict() for s in ("S1", "S2", "S3", "S4")}
    tables["mean_pbo_bootstrap"] = f"series-level percentile bootstrap, B={BOOT_B}, RandomState({BOOT_SEED})"
    dec = {"stage1": {"D1": d1(rates)}, "D2": "pending (Stage 2)"}
else:
    dec = json.load(open(os.path.join(D, "decision.json")))
    dec["D2"] = d2(rates)
    v = df[df.valid == 1]
    tables["mean_holdout_auc"] = {a: {h: float(v[v.arm_id == a][f"auc_{h}"].mean()) for h in ("u", "p")}
                                  for a in sorted(v.arm_id.unique())}
dec["prereg"] = {"file": "valid-framework experiments/inbox7/PREREGISTRATION.md", "commit": "bae70d3",
                 "sha256": "7ee7d405274d60f7cbda05989920b0f999baeb1730bd617726f98167a441fad2"}
dec["rules"] = {"size": "pass rate in B0", "power": "pass rate in B2",
                "test": "scipy.stats.fisher_exact([[k_B2, n_B2-k_B2], [k_B0, n_B0-k_B0]], alternative='greater')",
                "pbo_pass": "PBO < threshold (NaN = not passing)"}
json.dump(dec, open(os.path.join(D, "decision.json"), "w"), indent=1, default=float)
json.dump(tables, open(os.path.join(D, f"stage{STAGE}_tables.json"), "w"), indent=1, default=float)
print(json.dumps(dec, indent=1, default=float)[:6000])

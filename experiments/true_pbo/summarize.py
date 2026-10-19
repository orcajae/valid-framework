"""CHANGELOG E14: merge run parts into results/true_pbo/{null,power,real}.{csv,npz} and print the report tables.

usage: python3 -I experiments/true_pbo/summarize.py results/true_pbo
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
P = os.path.join(D, "parts")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    a = z / d * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, c - a), min(1.0, c + a)


def merge(prefix, name):
    parts = sorted(glob.glob(os.path.join(P, prefix + "*.csv")))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    df = df.sort_values(["asset", "beta", "seed"] if "beta" in df else ["asset", "seed"], na_position="first")
    df.to_csv(os.path.join(D, name + ".csv"), index=False)
    arrs = {}
    for f in parts:
        z = np.load(f[:-4] + ".npz")
        tag = os.path.basename(f)[:-4]
        arrs.update({f"{tag}__{k}": z[k] for k in z.files})
    np.savez_compressed(os.path.join(D, name + ".npz"), **arrs)
    return df


GATES = {
    "AUC>0.55": lambda d: d["auc"] > 0.55,
    "Perm": lambda d: d["perm_pass"].astype(bool),
    "PBO<0.20": lambda d: d["pbo_P"] < 0.20,
    "PBO<0.50": lambda d: d["pbo_P"] < 0.50,
    "NetSR>0": lambda d: d["net_sr_pe"] > 0,
}


def combos():
    names = list(GATES)
    out = [(n,) for n in names]
    for r in range(2, 5):
        for c in itertools.combinations(names, r):
            if "PBO<0.20" in c and "PBO<0.50" in c:   # PBO<0.20 implies PBO<0.50
                continue
            out.append(c)
    return out


def gate_table(df, group_cols):
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        g = g[g["valid"] == 1]
        n = len(g)
        for c in combos():
            m = np.ones(n, dtype=bool)
            for name in c:
                m &= GATES[name](g).values
            k = int(m.sum())
            lo, hi = wilson(k, n)
            rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                         "gate": " & ".join(c), "k": k, "n": n, "rate": k / n if n else np.nan,
                         "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


summary = {}
null = merge("null_", "null")
power = merge("power_", "power")
tabs = []
if null is not None:
    t = gate_table(null, ["asset"]); t.insert(0, "arm", "null"); tabs.append(t)
    summary["null_n"] = null.groupby("asset")["valid"].sum().to_dict()
    summary["null_auc_gt_055_pct"] = (null.groupby("asset")["auc"].apply(lambda s: round(100 * (s > 0.55).mean(), 1))).to_dict()
    summary["null_pbo_disagree_P"] = int(null["pbo_P_disagree"].sum())
    summary["null_pbo_disagree_R"] = int(null["pbo_R_disagree"].sum())
    vr = null.groupby("asset")["var_sr_is_R"].describe(percentiles=[0.5, 0.95, 0.99])
    summary["null_var_sr_is_R"] = vr.round(6).to_dict(orient="index")
    summary["null_pbo_R_lt_020_pct"] = null.groupby("asset")["pbo_R"].apply(lambda s: round(100 * (s < 0.2).mean(), 1)).to_dict()
if power is not None:
    t = gate_table(power, ["beta"]); t.insert(0, "arm", "power"); tabs.append(t)
    summary["power_mean_auc"] = power.groupby("beta")["auc"].mean().round(4).to_dict()
    summary["power_pbo_disagree_P"] = int(power["pbo_P_disagree"].sum())
if tabs:
    pd.concat(tabs).to_csv(os.path.join(D, "gate_rates.csv"), index=False)
real_p = os.path.join(P, "real.csv")
if os.path.exists(real_p):
    real = pd.read_csv(real_p)
    real.to_csv(os.path.join(D, "real.csv"), index=False)
    z = np.load(os.path.join(P, "real.npz"))
    np.savez_compressed(os.path.join(D, "real.npz"), **{k: z[k] for k in z.files})
    summary["real"] = json.loads(real.to_json(orient="records"))
    if null is not None:
        r1 = real.loc[real["setting"] == "round_1h_gridR", "var_sr_is_R"].iloc[0]
        summary["R1_var_sr_is_R"] = r1
        summary["R1_share_of_null_le_real"] = null.groupby("asset")["var_sr_is_R"].apply(lambda s: round(float((s <= r1).mean()), 4)).to_dict()
print(json.dumps(summary, indent=1, default=float))
json.dump(summary, open(os.path.join(D, "tables.json"), "w"), indent=1, default=float)

"""Corpus definitions for CHANGELOG E16. Reads results/reference/variants_340.csv only.

usage (python3 -I, from the repository root):
  python3 -I scripts/corpus_definitions.py [CSV [OUT_DIR]]
  CSV defaults to results/reference/variants_340.csv.

Prints and, with OUT_DIR, writes:
  corpus_definitions.csv  per definition: rows, negative net Sharpe in net_sr_18bp, share,
                          ladder rungs, Deflated Sharpe Ratio benchmark / best SR / best DSR / passes
  corpus_rows.csv         per row of variants_340.csv: kind, rule or model, membership per definition

Ladder rows are the rows scripts/derive_tf_ladder.py counts (COST_, BASE_, MC_, XS_ and SHAP_
excluded). They include the eight FORT_ rows, which are rule-based (a realised-volatility filter on
the momentum benchmark, no model fit); the daily rung without them is printed separately. The DSR follows the multiple-testing script of E13, with N = rows of the
definition and T = events (holdout = the last 20%).
"""
import math
import os
import statistics
import sys
from statistics import NormalDist

import pandas as pd

ND = NormalDist()
ANN = math.sqrt(252)
GAMMA = 0.5772156649
T_DEFAULT = {"1d": 864, "1h": 15271, "4h": 3818, "15m": 61084, "multi": 864}
STORED = ["net_sr_18bp", "auc", "unbal_long_pct", "bal_long_pct", "events", "asset", "model"]
LADDER_EXCLUDED = ("COST_", "BASE_", "MC_", "XS_", "SHAP_")
RULE = ("BASE_", "SHAP_", "XS_", "FORT_", "COST_benchmark")
RULE_IN_LADDER = ("FORT_",)
SYNTHETIC_18 = ["COST_ML_1h_18bp", "COST_ML_15m_18bp", "BASE_1h ML balanced"]
# rows that repeat another row's result under a different label
LABEL_DUPLICATES = ["BB_BTC_CatBoost", "BB_ETH_CatBoost", "BB_SOL_CatBoost", "COST_benchmark_18bp"]


def cost_other(v):
    return v.strategy_id.str.startswith("COST_") & ~v.strategy_id.str.endswith("_18bp")


def ablation(v):
    return v.strategy_id.str.startswith("ABLATION_")


def holdout_t(d):
    return [int(e) if e and e == e and e > 0 else T_DEFAULT[tf] for e, tf in zip(d.events, d.timeframe)]


def psr(s, t, sr0):
    n = max(t - int(t * 0.8) - 1, 1)
    return ND.cdf((s - sr0) * math.sqrt(n) / math.sqrt(1 + 0.5 * s * s))


def dsr(d):
    n = len(d)
    em = (1 - GAMMA) * ND.inv_cdf(1 - 1 / n) + GAMMA * ND.inv_cdf(1 - 1 / (n * math.e))
    s = [x / ANN for x in d.net_sr_18bp]
    sr0 = statistics.pstdev(s) * em
    p = [psr(si, ti, sr0) for si, ti in zip(s, holdout_t(d))]
    return round(sr0 * ANN, 2), round(max(d.net_sr_18bp), 2), round(max(p), 2), sum(x > 0.95 for x in p)


def ladder(d, exclude=LADDER_EXCLUDED):
    k = d[~d.strategy_id.str.startswith(exclude) & d.timeframe.isin(["15m", "1h", "4h", "1d"])]
    return {tf: f"{int((g.net_sr_18bp < 0).sum())}/{len(g)}" for tf, g in k.groupby("timeframe")}


def corrected(v):
    d = v[~cost_other(v) & ~v.strategy_id.isin(SYNTHETIC_18) & ~ablation(v)]
    d = d.drop_duplicates(STORED)
    return d[~d.strategy_id.isin(LABEL_DUPLICATES)]


def definitions(v):
    no_cost_rows = v[~v.strategy_id.str.startswith("COST_")]
    return {
        "as_printed": v,
        "no_cost_rows": no_cost_rows,
        "no_cost_rows_equal_rows_merged": no_cost_rows.drop_duplicates(STORED + ["timeframe"]),
        "corrected": corrected(v),
    }


def sensitivity(v):
    c = corrected(v)
    keep_abl = v[~cost_other(v) & ~v.strategy_id.isin(SYNTHETIC_18)
                 & ~(ablation(v) & v.strategy_id.isin(["ABLATION_No_Cost", "ABLATION_No_Cost_No_Perm", "ABLATION_AUC_Only"]))]
    keep_abl = keep_abl.drop_duplicates(STORED + ["timeframe"])
    keep_abl = keep_abl[~keep_abl.strategy_id.isin(LABEL_DUPLICATES)]
    early = v[~cost_other(v) & ~v.strategy_id.isin(SYNTHETIC_18) & ~ablation(v)].drop_duplicates(STORED)
    return {
        "corrected_keeping_ablation_results_with_costs": keep_abl,
        "corrected_without_baselines": c[~c.strategy_id.str.startswith("BASE_")],
        "corrected_without_fort_rows": c[~c.strategy_id.str.startswith("FORT_")],
        "earlier_version_label_duplicates_kept": early,
        "earlier_version_benchmark_row_kept": early[~early.strategy_id.isin(LABEL_DUPLICATES[:3])],
    }


def kind(sid):
    if sid.startswith("COST_") and not sid.endswith("_18bp"):
        return "cost level other than 18 bp"
    if sid in SYNTHETIC_18:
        return "synthetic signal at 18 bp"
    if sid.startswith("ABLATION_"):
        return "ablation refit"
    return "strategy variant"


def group(sid):
    if sid.startswith(RULE):
        return "rule"
    return "model"


def rows_table(v):
    """One line per row: kind, group, which earlier row it repeats, membership per definition."""
    base = v[~cost_other(v) & ~v.strategy_id.isin(SYNTHETIC_18) & ~ablation(v)]
    first_of = {}
    for sid, key in zip(base.strategy_id, map(tuple, base[STORED].astype(str).values)):
        first_of.setdefault(key, sid)
    exact = {sid: first_of[key] for sid, key in zip(base.strategy_id, map(tuple, base[STORED].astype(str).values))
             if first_of[key] != sid}
    label = {"BB_BTC_CatBoost": "EXT_BTC_1h_CatBoost", "BB_ETH_CatBoost": "EXT_ETH_1h_CatBoost",
             "BB_SOL_CatBoost": "EXT_SOL_1h_CatBoost", "COST_benchmark_18bp": "BASE_momentum_benchmark"}
    status, dup = [], []
    for sid in v.strategy_id:
        k = kind(sid)
        if k != "strategy variant":
            status.append(k); dup.append("")
        elif sid in exact:
            status.append("same result, every stored value equal"); dup.append(exact[sid])
        elif sid in label:
            status.append("same result, different label"); dup.append(label[sid])
        else:
            status.append("counted"); dup.append("")
    t = pd.DataFrame({"strategy_id": v.strategy_id, "family": v.strategy_id.str.extract(r"^([A-Za-z]+)")[0],
                      "net_sr_18bp": v.net_sr_18bp, "status": status, "duplicate_of": dup,
                      "group": [group(x) if kind(x) == "strategy variant" else "not a variant" for x in v.strategy_id]})
    for k, d in definitions(v).items():
        t["in_" + k] = v.strategy_id.isin(d.strategy_id)
    return t


def row(name, d):
    neg = int((d.net_sr_18bp < 0).sum())
    b, m, x, p = dsr(d)
    lad = ladder(d)
    return {"definition": name, "rows": len(d), "negative": neg, "share_pct": round(100 * neg / len(d), 1),
            "dsr_benchmark": b, "best_sr": m, "best_dsr": x, "dsr_passes": p,
            "ladder_15m": lad.get("15m"), "ladder_1h": lad.get("1h"), "ladder_4h": lad.get("4h"),
            "ladder_1d": lad.get("1d")}


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "reference", "variants_340.csv")
    v = pd.read_csv(src, keep_default_na=False, na_values=[""])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    defs = definitions(v)
    table = pd.DataFrame([row(k, d) for k, d in defs.items()])
    print(table.to_csv(index=False), end="")
    print("\nsensitivity:")
    print(pd.DataFrame([row(k, d) for k, d in sensitivity(v).items()]).to_csv(index=False), end="")

    c = defs["corrected"]
    print("\nrows that are not, or not separately, variants at 18 bp (rows, negative):")
    rows_tab = rows_table(v)
    for k in ["cost level other than 18 bp", "synthetic signal at 18 bp", "ablation refit",
              "same result, every stored value equal", "same result, different label"]:
        g = rows_tab[rows_tab.status == k]
        print(f"  {k}: {len(g)}, {int((g.net_sr_18bp < 0).sum())}")
    print("\ncorrected, by group (rows, negative):")
    grp = rows_tab[rows_tab.in_corrected]
    for k in ["rule", "model"]:
        g = grp[grp.group == k]
        print(f"  {k}: {len(g)}, {int((g.net_sr_18bp < 0).sum())}", {k2: int(n2) for k2, n2 in g.family.value_counts().items()} if k != "model" else "")
    print("corrected, ladder without the FORT_ rule rows:", ladder(c, LADDER_EXCLUDED + RULE_IN_LADDER))
    dl = v[v.strategy_id.str.startswith("DL_") | v.strategy_id.isin(["BB_LSTM_unbalanced", "BB_SimpleRNN_unbalanced"])]
    arch = dl.model.replace({"LSTM": "LSTM (BB_ row)", "SimpleRNN": "SimpleRNN (BB_ row)"})
    means = dl.groupby(arch).unbal_long_pct.agg(["size", "mean"]).sort_values("mean")
    print("deep learning, unweighted long share: per-architecture mean (rows)")
    for a2, r2 in means.iterrows():
        print(f"  {a2}: {100 * r2['mean']:.1f}% ({int(r2['size'])})")
    print(f"  single rows: {100 * dl.unbal_long_pct.min():.1f}-{100 * dl.unbal_long_pct.max():.1f}%")
    removed = v[~v.strategy_id.isin(c.strategy_id)]
    print(f"removed from 340: {len(removed)} rows, {int((removed.net_sr_18bp < 0).sum())} negative")
    s = [x / ANN for x in v.net_sr_18bp]
    p0 = [psr(si, ti, 0.0) for si, ti in zip(s, holdout_t(v))]
    i = max(range(len(p0)), key=p0.__getitem__)
    print(f"DSR formula against a benchmark of 0: highest {p0[i]:.2f} ({v.strategy_id.iloc[i]}, "
          f"{holdout_t(v)[i] - int(holdout_t(v)[i] * 0.8)} holdout events); rows above 0.95: {sum(x > 0.95 for x in p0)}")
    e = v[v.strategy_id.str.match(r"^EXT_\w+_1h_")]
    print("bull bias, nine 1h tree rows (EXT_*_1h_*), by model: unweighted long share / weighted / AUC")
    for mdl, g in e.groupby("model"):
        print(f"  {mdl}: {g.unbal_long_pct.min():.4f}-{g.unbal_long_pct.max():.4f} / "
              f"{g.bal_long_pct.min():.4f}-{g.bal_long_pct.max():.4f} / {g.auc.min():.4f}-{g.auc.max():.4f}")
    tr = v[v.model.isin(["CatBoost", "LightGBM", "RF"]) & v.unbal_long_pct.notna()]
    print(f"  all tree rows with a long share: {len(tr)}, unweighted {tr.unbal_long_pct.min():.4f}-{tr.unbal_long_pct.max():.4f}")
    dl = v[v.strategy_id.str.startswith("DL_")]
    mlp = dl[dl.model.str.startswith("MLP")]
    print(f"  DL_ rows: {len(dl)}, unweighted {dl.unbal_long_pct.min():.4f}-{dl.unbal_long_pct.max():.4f}; "
          f"MLP rows with weighted = unweighted: {int((mlp.bal_long_pct == mlp.unbal_long_pct).sum())} of {len(mlp)}")
    a = v[ablation(v)]
    print(f"ablation: {len(a)} rows, {len(a.drop_duplicates(['net_sr_18bp', 'auc', 'bal_long_pct', 'unbal_long_pct']))} "
          f"distinct results, {len(a.drop_duplicates(['model', 'auc', 'bal_long_pct', 'unbal_long_pct']))} fitted models")

    if out:
        os.makedirs(out, exist_ok=True)
        table.to_csv(os.path.join(out, "corpus_definitions.csv"), index=False)
        rows = rows_table(v)
        rows.to_csv(os.path.join(out, "corpus_rows.csv"), index=False)

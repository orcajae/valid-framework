"""Size and power of PBO thresholds, and a calibrated test of rank persistence: merge parts, write the rates
and the mechanical decision file.

Design: experiments/cscv_calibration/PREREGISTRATION.md (2ead385), section 4 (D1-D5 and "Use of the
outcomes"). Every verdict follows section 4 literally; nothing is chosen here. The rule functions take
counts only, so --selftest exercises every branch on fabricated toy counts. numpy, scipy and pandas only.

usage (python3 -I, from the repository root):
  summarize.py [OUT_DIR]        # default <repo>/results/cscv_calibration (relative to this file); reads OUT_DIR/parts/
  summarize.py --selftest       # rule branches on clearly synthetic toy counts; writes nothing
  summarize.py DEV_DIR --dev-n N    # dev-namespace smoke runs only: expect min(N, registered) series per unit
Writes to OUT_DIR:
  stage{1,2,3}_rows.csv         merged per-series rows (PBO stored as counts k of K splits)
  stage2_star.npz, stage3_star.npz   PBO* counts per resample; stage3_R.npz  T x 24 return matrices
  rates_layer1.csv              per cell x arm x tie rule x tau: PBO < tau, Wilson 95%
  means_layer1.csv              per cell x arm: mean PBO (3 rules), mean share of splits with a tie, mean best_pop_sr
  rates_stage2.csv              per cell x arm: calibrated p <= 0.05 (L 20; L 5, 50 where run), Reality Check, mean p
  rates_layer2.csv              per arm x (M, S): PBO < tau (3 rules), calibrated p, Reality Check, mean PBO, mean p
  d5_table.csv                  D5: Layer 2 rates beside the nearest Layer 1 cells (descriptive)
  decision.json                 D1-D4, Use of the outcomes, completeness (dropped F arms with their reason from
                                beta_F.json), the prereg body digest and this file's sha256
Stops (SystemExit) on: mixed namespaces, mixed B within a stage, a Stage 2 observed PBO that differs from
Stage 1, a prereg body that differs from 2ead385, unpaired D4 series.
Conventions (also in decision.json "rules"):
  - pass of a PBO screen: k/K < tau; pass of a p-value: 20 * (1 + count) <= B + 1 (p <= 0.05, integer form)
  - Layer 2 series with valid = 0 count as not passing every screen (prereg section 5); denominators include them
  - percentage-point comparisons in D2-D4 use exact fractions; D1 uses counts per 1,000 series
  - D1 R95: numpy default_rng(0).binomial(1000, p_bar, size=(10000, 216)) / 1000, span per row,
    np.percentile(spans, 95) (linear interpolation)
  - D2: scipy fisher_exact([[k_N2, n-k_N2], [k_N1, n-k_N1]], alternative='greater')
  - D4: exact two-sided McNemar, p = min(1, 2 * Binomial(b + c, 1/2).cdf(min(b, c))), p = 1 if b + c = 0
  - Wilson interval as E15 (z = 1.96)
"""
import glob
import hashlib
import json
import os
import sys
from fractions import Fraction
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact

TS, MS, SS, RHOS, SERIALS = (250, 1000, 2500), (4, 8, 16, 32), (8, 16), (0.0, 0.5, 0.9), ("IID", "GARCH", "MA10")
CELLS = list(product(TS, MS, SS, RHOS, SERIALS))
REF, UNEQ = CELLS.index((1000, 8, 16, 0.5, "GARCH")), 216
REPS1 = {"N1": 1000, "N2": 1000, "N3": 500, "P1": 500, "P2": 500, "P3": 500}
UNEQ_REPS = {"N1": 1000, "N2": 1000, "P2": 500}
S2_CELLS = [c for c, (T, M, S, rho, _) in enumerate(CELLS)
            if T in (250, 1000) and M in (4, 8, 16) and S == 8 and rho == 0.5] + [REF]
S2_ARMS, S2_REPS = ("N1", "N2", "N3", "P2"), 500
TAUS = (0.05, 0.10, 0.20, 0.50)
RULES = {"first": "k_first", "last": "k_last", "mean_rank": "k_mean"}
L2_ARMS = ("A-BTC", "A-ETH", "A-SOL", "D", "C", "B0", "B1", "B2", "F1", "F2", "R")
L2_N = {**{a: 200 for a in L2_ARMS[:5]}, **{a: 100 for a in L2_ARMS[5:10]}, "R": 1}
L2_M, CALIB_MS = (4, 8, 16, 24), ((8, 16), (16, 8))
D1_DRAWS, D1_SEED, D1_N = 10000, 0, 1000
CONVENTIONS = {
    "pbo_screen": "pass if k/K < tau (PBO under the first-listed tie rule unless a rule is named)",
    "p_value": "pass if 20 * (1 + count) <= B + 1, i.e. p = (1 + count) / (B + 1) <= 0.05",
    "layer2_invalid": "series with valid = 0 count as not passing every screen; denominators include them",
    "percentage_points": "D2-D4 compare exact fractions; D1 works in counts per n series",
    "D1_R95": "default_rng(0).binomial(n, p_bar, size=(10000, 216)) / n; span per row; np.percentile(spans, 95)",
    "D2_test": "scipy.stats.fisher_exact([[k_N2, n-k_N2], [k_N1, n-k_N1]], alternative='greater')",
    "D4_test": "exact two-sided McNemar: min(1, 2 * Binomial(b + c, 1/2).cdf(min(b, c))); 1 if b + c = 0",
    "wilson": "Wilson score interval, z = 1.96 (as E15)",
    "layer2_ties": "per arm x (M, S): mean share of splits with an in-sample tie, series with a tie, mean PBO per rule",
    "dropped_F_arms": "F arms dropped at calibration are reported with the reason from beta_F.json, not as missing",
    "stops": "mixed namespaces or B within a stage, Stage 2 observed PBO != Stage 1, prereg body changed, unpaired D4",
}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(HERE, "..", "..", "results", "cscv_calibration")
PREREG_PATH = os.path.join(HERE, "PREREGISTRATION.md")
PREREG_BODY_SHA = "cf81e2099a6cdd08c831df0df0e0274838c6c0e0af3fd599db626bf92d7dd73a"   # text before "## Change log"
PREREG = {"file": "valid-framework experiments/cscv_calibration/PREREGISTRATION.md", "commit": "2ead385",
          "body_sha256": PREREG_BODY_SHA}
F_ARMS = ("F1", "F2")


def need(cond, msg):
    """Explicit check (survives python -O)."""
    if not cond:
        raise SystemExit(f"STOP: {msg}")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prereg_body():
    t = open(PREREG_PATH, encoding="utf-8").read()
    body = hashlib.sha256(t[:t.index("## Change log")].encode()).hexdigest()
    need(body == PREREG_BODY_SHA, "PREREGISTRATION.md (before the Change log) differs from 2ead385")
    return {**PREREG, "file_sha256_now": sha256_file(PREREG_PATH)}


def dropped_arms(out, ns):
    """F arms dropped at calibration, with the reason recorded in beta_F.json."""
    p = os.path.join(out, "beta_F.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    need(d.get("ns") == ns, f"{p} namespace {d.get('ns')} differs from the parts ({ns})")
    return {a: ((d.get(a) or {}).get("reason") or "dropped") for a in F_ARMS
            if d.get(a) is None or (isinstance(d[a], dict) and d[a].get("dropped"))}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    a = z / d * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, c - a), min(1.0, c + a)


def p_pass(count, B):
    """p = (1 + count) / (B + 1) <= 0.05, in integers."""
    return 20 * (1 + np.asarray(count)) <= np.asarray(B) + 1


def pp(k, n):
    return Fraction(100 * int(k), int(n))


# ---------------------------------------------------------------- decision rules (counts in, verdicts out)
def d1_verdict(excess_pp):
    if excess_pp > 5:
        return "design-dependent"
    if excess_pp <= 2:
        return "design-invariant"
    return "intermediate"


def d1_rule(ks, n=D1_N):
    """ks: N1 pass counts of the 216 cells, n series each."""
    ks = np.asarray(ks, dtype=np.int64)
    pbar = ks.sum() / (n * len(ks))
    sims = np.random.default_rng(D1_SEED).binomial(n, pbar, size=(D1_DRAWS, len(ks)))
    r95 = float(np.percentile(sims.max(1) - sims.min(1), 95))         # in counts per n
    span = int(ks.max() - ks.min())
    excess_pp = (span - r95) * 100 / n
    return {"span_pp": span * 100 / n, "pooled_rate": float(pbar), "R95_pp": r95 * 100 / n,
            "span_minus_R95_pp": excess_pp, "min_cell": int(ks.argmin()), "max_cell": int(ks.argmax()),
            "verdict": d1_verdict(excess_pp)}


def d2_rule(k1, n1, k2, n2):
    diff = pp(k2, n2) - pp(k1, n1)
    p = float(fisher_exact([[k2, n2 - k2], [k1, n1 - k1]], alternative="greater")[1])
    if diff >= 10 and p < 0.05:
        v = "responds"
    elif diff < 3:
        v = "does not respond"
    else:
        v = "unclear"
    return {"N1": [int(k1), int(n1)], "N2": [int(k2), int(n2)], "N2_minus_N1_pp": float(diff), "p_F": p, "verdict": v}


def d3_rule(n1_cells, l2_a, p2_ref):
    """n1_cells: {cell: (k, n)} for the 19 Stage 2 cells; l2_a: {(M, S): (k, n)} pooled arm A; p2_ref: (k, n)."""
    each = {c: bool(Fraction(int(k), int(n)) <= Fraction(8, 100)) for c, (k, n) in n1_cells.items()}
    K, N = sum(k for k, _ in n1_cells.values()), sum(n for _, n in n1_cells.values())
    hi = wilson(K, N)[1]
    l2 = {f"M{m}_S{s}": bool(Fraction(int(k), int(n)) <= Fraction(8, 100)) for (m, s), (k, n) in l2_a.items()}
    size = all(each.values()) and hi <= 0.065 and all(l2.values()) and len(l2) == 2 and len(each) == 19
    power = Fraction(int(p2_ref[0]), int(p2_ref[1])) >= Fraction(1, 2)
    v = "fails" if not size else ("works" if power else "size only")
    return {"each_cell_le_8pct": each, "all_cells_le_8pct": all(each.values()), "pooled_k": int(K), "pooled_n": int(N),
            "pooled_wilson_hi": hi, "pooled_hi_le_6.5pct": bool(hi <= 0.065),
            "layer2_A": {f"M{m}_S{s}": {"k": int(k), "n": int(n), "le_8pct": l2[f"M{m}_S{s}"]} for (m, s), (k, n) in l2_a.items()},
            "size_controlled": bool(size), "P2_ref": [int(p2_ref[0]), int(p2_ref[1])], "power": bool(power), "verdict": v}


def mcnemar_exact(b, c):
    m = b + c
    return 1.0 if m == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), m, 0.5)))


def d4_rule(pass_p2, pass_n3, what="PBO"):
    a, b = np.asarray(pass_p2, bool), np.asarray(pass_n3, bool)
    need(len(a) == len(b), "D4: unpaired series")
    n = len(a)
    bb, cc = int(np.sum(a & ~b)), int(np.sum(~a & b))
    diff = pp(a.sum(), n) - pp(b.sum(), n)
    p = mcnemar_exact(bb, cc)
    if diff >= 10 and p < 0.05:
        v = f"{what} separates edge from spread"
    elif abs(diff) < 5:
        v = f"{what} does not separate them"
    else:
        v = "unclear" + (" (N3 - P2 >= 10 percentage points)" if -diff >= 10 else "")
    return {"pairs": n, "P2_pass": int(a.sum()), "N3_pass": int(b.sum()), "P2_only": bb, "N3_only": cc,
            "P2_minus_N3_pp": float(diff), "p_Q4": p, "verdict": v}


def outcome_rule(d1_020, d1_005, d3, d2=None):
    if d3 == "works":
        return {"row": 1, "paper": "methods paper: calibrated test of rank persistence under H0 'equal means' (size, "
                "power, N2 and N3 pass rates as rates of passing sets without a positive edge, D1, D4, Reality Check)"}
    if "design-dependent" in (d1_020, d1_005):
        return {"row": 2, "paper": "short paper on the size and power of PBO thresholds; D2 "
                + ("presented as the mechanism" if d2 == "responds" else "reported without a mechanism claim") + "; D4 reported"}
    if d1_020 == "design-invariant" and d1_005 == "design-invariant":
        return {"row": 3, "paper": "short paper with a threshold table for practitioners, with D2 and D4"}
    return {"row": 4, "paper": "no paper; results published as a changelog entry only"}


# ---------------------------------------------------------------- merge
def read_parts(out, stage):
    files = sorted(glob.glob(os.path.join(out, "parts", f"s{stage}", "*.csv")))
    if not files:
        return None, files
    df = pd.concat([pd.read_csv(f, float_precision="round_trip", keep_default_na=False, na_values=[""])
                    for f in files], ignore_index=True)
    return df, files


def cell_cols(df):
    p = np.array([CELLS[REF if c == UNEQ else c] for c in df["cell"]], dtype=object)
    for i, k in enumerate(("T", "M", "S", "rho", "serial")):
        df[k] = p[:, i]
    df["uneq"] = df["cell"] == UNEQ
    return df


def layer1(df):
    rates, means = [], []
    for (c, arm), g in df.groupby(["cell", "arm"], sort=True):
        n = len(g)
        T, M, S, rho, ser = CELLS[REF if c == UNEQ else c]
        key = {"cell": c, "T": T, "M": M, "S": S, "rho": rho, "serial": ser, "uneq": c == UNEQ, "arm": arm}
        mrow = dict(key, n=n, mean_tie_share=float((g.k_tie / g.K).mean()), mean_best_pop_sr=float(g.best_pop_sr.mean()))
        for rule, col in RULES.items():
            pbo = g[col] / g.K
            mrow[f"mean_pbo_{rule}"] = float(pbo.mean())
            for tau in TAUS:
                k = int((pbo < tau).sum())
                lo, hi = wilson(k, n)
                rates.append(dict(key, rule=rule, statistic=f"PBO<{tau:.2f}", k=k, n=n, rate=k / n, wilson_lo=lo, wilson_hi=hi))
        means.append(mrow)
    return pd.DataFrame(rates), pd.DataFrame(means)


def complete1(df, cap):
    have = df.groupby(["cell", "arm"]).j.nunique().to_dict()
    need = {(c, a): min(n, cap) for c in range(UNEQ) for a, n in REPS1.items()}
    need.update({(UNEQ, a): min(n, cap) for a, n in UNEQ_REPS.items()})
    return [f"c{c}_{a}" for (c, a), n in need.items() if have.get((c, a), 0) != n]


def stage2(df):
    rows = []
    for (c, arm), g in df.groupby(["cell", "arm"], sort=True):
        n = len(g)
        T, M, S, rho, ser = CELLS[c]
        key = {"cell": c, "T": T, "M": M, "S": S, "rho": rho, "serial": ser, "arm": arm, "B": int(g.B.iloc[0])}
        stats = {"calibrated_p<=0.05_L20": (p_pass(g.le_L20, g.B), g.p_L20),
                 "reality_check_p<=0.05": (p_pass(g.ge_rc, g.B), g.p_rc)}
        for L in (5, 50):
            if f"le_L{L}" in g and g[f"le_L{L}"].notna().all():
                stats[f"calibrated_p<=0.05_L{L}"] = (p_pass(g[f"le_L{L}"].astype(int), g.B), g[f"p_L{L}"])
        for name, (m, p) in stats.items():
            k = int(m.sum())
            lo, hi = wilson(k, n)
            rows.append(dict(key, statistic=name, k=k, n=n, rate=k / n, wilson_lo=lo, wilson_hi=hi, mean_p=float(p.mean())))
    return pd.DataFrame(rows)


def layer2(df):
    rows = []
    groups = [(a, df[df.arm == a]) for a in L2_ARMS if (df.arm == a).any()]
    a_all = df[df.arm.isin(["A-BTC", "A-ETH", "A-SOL"])]
    if len(a_all):
        groups.append(("A (pooled)", a_all))
    for arm, g in groups:
        n, v = len(g), g[g.valid == 1]
        for M in L2_M:
            for S in SS:
                base = {"arm": arm, "M": M, "S": S, "n": n, "n_invalid": int((g.valid != 1).sum())}
                tie = v[f"M{M}_S{S}_k_tie"] / v[f"M{M}_S{S}_K"]
                kt = int((tie > 0).sum())
                rows.append(dict(base, rule="", statistic="mean_share_of_splits_with_IS_tie",
                                 mean=float(tie.mean()) if len(v) else np.nan))
                rows.append(dict(base, rule="", statistic="series_with_an_IS_tie", k=kt, rate=kt / n))
                for rule, col in RULES.items():
                    pbo = v[f"M{M}_S{S}_{col}"] / v[f"M{M}_S{S}_K"]
                    rows.append(dict(base, rule=rule, statistic="mean_pbo", mean=float(pbo.mean()) if len(v) else np.nan))
                    for tau in TAUS:
                        k = int((pbo < tau).sum())
                        lo, hi = wilson(k, n)
                        rows.append(dict(base, rule=rule, statistic=f"PBO<{tau:.2f}", k=k, rate=k / n, wilson_lo=lo,
                                         wilson_hi=hi, mean=float(pbo.mean()) if len(v) else np.nan))
                if (M, S) in CALIB_MS:
                    pre = f"cal_M{M}_S{S}_"
                    stats = {"calibrated_p<=0.05_L20": ("le_L20", "p_L20"), "reality_check_p<=0.05": ("ge_rc", "p_rc")}
                    stats.update({f"calibrated_p<=0.05_L{L}": (f"le_L{L}", f"p_L{L}") for L in (5, 50)
                                  if pre + f"le_L{L}" in v and len(v) and v[pre + f"le_L{L}"].notna().all()})
                    for name, (cnt, pc) in stats.items():
                        k = int(p_pass(v[pre + cnt].astype(int), v.B.astype(int)).sum()) if len(v) else 0
                        lo, hi = wilson(k, n)
                        rows.append(dict(base, rule="first", statistic=name, k=k, rate=k / n, wilson_lo=lo, wilson_hi=hi,
                                         mean=float(v[pre + pc].mean()) if len(v) else np.nan))
            if len(v):
                rows.append(dict(arm=arm, M=M, S=np.nan, n=n, n_invalid=int((g.valid != 1).sum()), rule="",
                                 statistic="mean_best_sr_pe", mean=float(v[f"M{M}_best_sr_pe"].mean())))
                rows.append(dict(arm=arm, M=M, S=np.nan, n=n, n_invalid=int((g.valid != 1).sum()), rule="",
                                 statistic="mean_median_pairwise_corr", mean=float(v[f"M{M}_med_corr"].mean())))
    return pd.DataFrame(rows)


def d5(r1, r2, rl2):
    """Layer 2 rates beside the nearest Layer 1 cells: T = 1000, rho = 0.5, each serial structure; M = 24 beside 16 and 32."""
    out = []
    pick = lambda d, **kw: d.loc[np.logical_and.reduce([d[k] == v for k, v in kw.items()])]  # noqa: E731
    for M in L2_M:
        for S in SS:
            stats = ["PBO<0.20"] + (["calibrated_p<=0.05_L20"] if (M, S) in CALIB_MS else [])
            for st in stats:
                for _, r in pick(rl2, M=M, S=S, rule="first", statistic=st).iterrows():
                    out.append({"block": "layer2", "layer2_M": M, "S": S, "arm": r.arm, "serial": "", "M_layer1": "",
                                "statistic": st, "k": r.k, "n": r.n, "rate": r.rate, "wilson_lo": r.wilson_lo,
                                "wilson_hi": r.wilson_hi})
            for m1 in ((16, 32) if M == 24 else (M,)):
                for ser in SERIALS:
                    if r1 is not None:
                        for _, r in pick(r1, T=1000, rho=0.5, M=m1, S=S, serial=ser, uneq=False, rule="first",
                                         statistic="PBO<0.20").iterrows():
                            out.append({"block": "layer1_nearest", "layer2_M": M, "S": S, "arm": r.arm, "serial": ser,
                                        "M_layer1": m1, "statistic": "PBO<0.20", "k": r.k, "n": r.n, "rate": r.rate,
                                        "wilson_lo": r.wilson_lo, "wilson_hi": r.wilson_hi})
                    if r2 is not None:
                        for _, r in pick(r2, T=1000, rho=0.5, M=m1, S=S, serial=ser,
                                         statistic="calibrated_p<=0.05_L20").iterrows():
                            out.append({"block": "layer1_nearest", "layer2_M": M, "S": S, "arm": r.arm, "serial": ser,
                                        "M_layer1": m1, "statistic": r.statistic, "k": r.k, "n": r.n, "rate": r.rate,
                                        "wilson_lo": r.wilson_lo, "wilson_hi": r.wilson_hi})
    return pd.DataFrame(out)


def main(out, dev_n=None):
    dec = {"prereg": prereg_body(), "summarize_py_sha256": sha256_file(os.path.abspath(__file__)),
           "completeness": {}, "consistency": {}}
    cap = dev_n or 10 ** 9
    s1, f1 = read_parts(out, 1)
    s2, f2 = read_parts(out, 2)
    s3, f3 = read_parts(out, 3)
    ns = {d.ns.iloc[0] for d in (s1, s2, s3) if d is not None}
    need(all(d is None or d.ns.nunique() == 1 for d in (s1, s2, s3)) and len(ns) <= 1, "namespaces mixed")
    dec["ns"] = ns.pop() if ns else None
    need(dev_n is None or dec["ns"] == "dev", "--dev-n is for dev-namespace runs only")
    if dev_n:
        dec["dev_n"] = dev_n
    r1 = r2 = rl2 = None
    if s1 is not None:
        s1 = cell_cols(s1.sort_values(["cell", "arm", "j"]).reset_index(drop=True))
        s1.to_csv(os.path.join(out, "stage1_rows.csv"), index=False)
        r1, m1 = layer1(s1)
        r1.to_csv(os.path.join(out, "rates_layer1.csv"), index=False)
        m1.to_csv(os.path.join(out, "means_layer1.csv"), index=False)
        dec["completeness"]["stage1_missing_units"] = complete1(s1, cap)
    if s2 is not None:
        s2 = s2.sort_values(["cell", "arm", "j"]).reset_index(drop=True)
        s2.to_csv(os.path.join(out, "stage2_rows.csv"), index=False)
        r2 = stage2(s2)
        r2.to_csv(os.path.join(out, "rates_stage2.csv"), index=False)
        star = {}
        for f in f2:
            z = np.load(f[:-4] + ".npz")
            star.update({f"{os.path.basename(f)[:-4]}_{k}": z[k] for k in z.files})
        np.savez_compressed(os.path.join(out, "stage2_star.npz"), **star)
        have = s2.groupby(["cell", "arm"]).j.nunique().to_dict()
        dec["completeness"]["stage2_missing_units"] = [f"c{c}_{a}" for c in S2_CELLS for a in S2_ARMS
                                                       if have.get((c, a), 0) != min(S2_REPS, cap)]
        dec["consistency"]["stage2_B"] = sorted(int(b) for b in s2.B.unique())
        need(len(dec["consistency"]["stage2_B"]) == 1, f"Stage 2 parts mix B values {dec['consistency']['stage2_B']}")
        if s1 is not None:
            m = s2.merge(s1[["cell", "arm", "j", "K", "k_first"]], on=["cell", "arm", "j"], how="left")
            hit = m.k_first.notna()
            bad = int(((m.k_obs != m.k_first) | (m.K_x != m.K_y))[hit].sum())
            need(bad == 0, f"{bad} Stage 2 observed PBO values differ from Stage 1")
            dec["consistency"]["stage2_pbo_obs_eq_stage1"] = {"compared": int(hit.sum()), "mismatches": bad}
    if s3 is not None:
        s3 = s3.sort_values(["arm", "seed"]).reset_index(drop=True)
        s3.to_csv(os.path.join(out, "stage3_rows.csv"), index=False)
        rl2 = layer2(s3)
        rl2.to_csv(os.path.join(out, "rates_layer2.csv"), index=False)
        star, mats = {}, {}
        for f in f3:
            z = np.load(f[:-4] + ".npz")
            for k in z.files:
                (mats if k.endswith("_R") else star)[k] = z[k]
        np.savez_compressed(os.path.join(out, "stage3_star.npz"), **star)
        np.savez_compressed(os.path.join(out, "stage3_R.npz"), **mats)
        have = s3.groupby("arm").seed.nunique().to_dict()
        dropped = dropped_arms(out, dec["ns"])
        need(not any(have.get(a) for a in dropped), f"rows exist for dropped arms {sorted(dropped)}")
        want = {a: (have.get(a, 0) if dev_n else L2_N[a]) for a in L2_ARMS if a not in dropped}
        dec["completeness"]["stage3_dropped"] = dropped
        dec["completeness"]["stage3_missing"] = {a: want[a] - have.get(a, 0) for a in want
                                                 if have.get(a, 0) != want[a] or not want[a]}
        dec["consistency"]["stage3_B"] = sorted(int(b) for b in s3.B.dropna().unique())
        need(len(dec["consistency"]["stage3_B"]) <= 1, f"Stage 3 parts mix B values {dec['consistency']['stage3_B']}")
        dec["completeness"]["stage3_invalid"] = s3[s3.valid != 1].groupby("arm").size().astype(int).to_dict()
    if r1 is not None and r2 is not None:
        d5(r1, r2, rl2 if rl2 is not None else pd.DataFrame(columns=["M", "S", "rule", "statistic"])).to_csv(
            os.path.join(out, "d5_table.csv"), index=False)

    # D1 (Stage 1, all 216 cells, arm N1)
    def ok1(cells_arms):
        return s1 is not None and not any(u in dec["completeness"]["stage1_missing_units"] for u in cells_arms)
    if ok1([f"c{c}_N1" for c in range(UNEQ)]):
        n1 = s1[(s1.arm == "N1") & (s1.cell < UNEQ)]
        n = n1.groupby("cell").size()
        need(n.nunique() == 1 and len(n) == UNEQ, "D1: unequal N1 counts across cells")
        dec["D1"] = {}
        for tau in (0.20, 0.05):
            ks = (n1.k_first / n1.K < tau).groupby(n1.cell).sum().reindex(range(UNEQ)).to_numpy()
            dec["D1"][f"tau={tau:.2f}"] = d1_rule(ks, int(n.iloc[0]))
    else:
        dec["D1"] = "pending (Stage 1 incomplete)"
    # D2 (reference cell, PBO < 0.20, N2 vs N1, n = 1000 each)
    if ok1([f"c{REF}_N1", f"c{REF}_N2"]):
        g = s1[s1.cell == REF]
        k = {a: int((g[g.arm == a].k_first / g[g.arm == a].K < 0.20).sum()) for a in ("N1", "N2")}
        n = {a: int((g.arm == a).sum()) for a in ("N1", "N2")}
        dec["D2"] = d2_rule(k["N1"], n["N1"], k["N2"], n["N2"])
    else:
        dec["D2"] = "pending (Stage 1 incomplete)"
    # D4 (reference cell, 500 pairs, PBO < 0.20; the same comparison for the calibrated p)
    if ok1([f"c{REF}_P2", f"c{REF}_N3"]):
        g = s1[(s1.cell == REF) & s1.arm.isin(["P2", "N3"])].sort_values("j")
        pa = (g[g.arm == "P2"].set_index("j").pipe(lambda d: d.k_first / d.K) < 0.20).sort_index()
        pb = (g[g.arm == "N3"].set_index("j").pipe(lambda d: d.k_first / d.K) < 0.20).sort_index()
        need(pa.index.equals(pb.index), "D4: P2 and N3 series are not paired (Stage 1)")
        dec["D4"] = {"PBO<0.20": d4_rule(pa.values, pb.values)}
        if s2 is not None and not any(u in dec["completeness"]["stage2_missing_units"] for u in (f"c{REF}_P2", f"c{REF}_N3")):
            h = s2[s2.cell == REF]
            qa = h[h.arm == "P2"].set_index("j").sort_index()
            qb = h[h.arm == "N3"].set_index("j").sort_index()
            need(qa.index.equals(qb.index), "D4: P2 and N3 series are not paired (Stage 2)")
            dec["D4"]["calibrated_p<=0.05 (reported)"] = d4_rule(p_pass(qa.le_L20, qa.B), p_pass(qb.le_L20, qb.B),
                                                                     "the calibrated p-value")
    else:
        dec["D4"] = "pending (Stage 1 incomplete)"
    # D3 (Stage 2 N1 in 19 cells, Layer 2 arm A pooled at both (M, S), P2 in the reference cell)
    s2_ok = s2 is not None and not dec["completeness"]["stage2_missing_units"]
    s3_ok = s3 is not None and all(a not in dec["completeness"]["stage3_missing"] for a in ("A-BTC", "A-ETH", "A-SOL"))
    if s2_ok and s3_ok:
        n1c = {}
        for c in S2_CELLS:
            g = s2[(s2.cell == c) & (s2.arm == "N1")]
            n1c[c] = (int(p_pass(g.le_L20, g.B).sum()), len(g))
        a = s3[s3.arm.isin(["A-BTC", "A-ETH", "A-SOL"])]
        av = a[a.valid == 1]
        l2a = {(m, s): (int(p_pass(av[f"cal_M{m}_S{s}_le_L20"].astype(int), av.B.astype(int)).sum()), len(a))
               for m, s in CALIB_MS}
        g = s2[(s2.cell == REF) & (s2.arm == "P2")]
        dec["D3"] = d3_rule(n1c, l2a, (int(p_pass(g.le_L20, g.B).sum()), len(g)))
        beside = {}
        for arm in ("N2", "N3"):
            g = s2[(s2.arm == arm) & s2.cell.isin(S2_CELLS)]
            per = {int(c): [int(p_pass(h.le_L20, h.B).sum()), len(h)] for c, h in g.groupby("cell")}
            k, n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
            beside[arm] = {"per_cell": per, "pooled": [k, n, k / n, *wilson(k, n)], "reference_cell": per.get(REF)}
        dec["D3"]["beside_not_in_verdict_calibrated_p_pass_rates"] = beside
    else:
        dec["D3"] = "pending (Stage 2 or Stage 3 arm A incomplete)"
    d = lambda k: dec[k]["verdict"] if isinstance(dec[k], dict) and "verdict" in dec[k] else None  # noqa: E731
    if isinstance(dec["D1"], dict) and d("D3"):
        dec["use_of_outcomes"] = outcome_rule(dec["D1"]["tau=0.20"]["verdict"], dec["D1"]["tau=0.05"]["verdict"],
                                              d("D3"), d("D2"))
    else:
        dec["use_of_outcomes"] = "pending"
    dec["D5"] = "d5_table.csv (descriptive, no verdict)"
    dec["rules"] = CONVENTIONS
    with open(os.path.join(out, "decision.json"), "w") as f:
        json.dump(dec, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in dec.items() if k in ("ns", "completeness", "consistency")}, indent=1, default=str))
    print(f"wrote decision.json and rate tables to {out}")


# ---------------------------------------------------------------- self-test (fabricated toy counts)
def selftest():
    """Every branch of D1-D4 and the outcome table on clearly synthetic toy counts (not results)."""
    ok = {}
    # D1: boundary mapping, then three fabricated cell vectors
    ok["d1_map"] = ([d1_verdict(x) for x in (5.0, 5.0001, 2.0, 2.0001, -3.0)]
                    == ["intermediate", "design-dependent", "design-invariant", "intermediate", "design-invariant"])
    flat = d1_rule([100] * 216)
    ok["d1_flat_invariant"] = flat["verdict"] == "design-invariant" and flat["span_pp"] == 0
    wide = d1_rule(list(np.linspace(50, 300, 216).round().astype(int)))
    ok["d1_wide_dependent"] = wide["verdict"] == "design-dependent"
    base = d1_rule([100] * 215 + [100])
    found = None
    for x in range(100, 300):
        r = d1_rule([100] * 215 + [x])
        if r["verdict"] == "intermediate":
            found = r
            break
    ok["d1_intermediate_exists"] = found is not None and 2 < found["span_minus_R95_pp"] <= 5 and base["R95_pp"] > 0
    # D2
    ok["d2"] = ([d2_rule(*a)["verdict"] for a in ((100, 1000, 200, 1000), (100, 1000, 129, 1000), (100, 1000, 130, 1000),
                                                  (100, 1000, 150, 1000), (2, 20, 5, 20), (100, 1000, 200, 1000),
                                                  (200, 1000, 100, 1000))]
                == ["responds", "does not respond", "unclear", "unclear", "unclear", "responds", "does not respond"])
    ok["d2_boundary_10pp"] = d2_rule(100, 1000, 200, 1000)["N2_minus_N1_pp"] == 10.0
    # D3
    cells = {c: (25, 500) for c in range(19)}
    l2 = {(8, 16): (30, 600), (16, 8): (30, 600)}
    ok["d3_works"] = d3_rule(cells, l2, (250, 500))["verdict"] == "works"
    ok["d3_size_only"] = d3_rule(cells, l2, (249, 500))["verdict"] == "size only"
    ok["d3_cell_8pct_ok"] = d3_rule({**cells, 0: (40, 500)}, l2, (300, 500))["verdict"] == "works"
    ok["d3_cell_over"] = d3_rule({**cells, 0: (41, 500)}, l2, (300, 500))["verdict"] == "fails"
    hi = d3_rule({c: (32, 500) for c in range(19)}, l2, (300, 500))
    ok["d3_pooled_over"] = hi["verdict"] == "fails" and hi["all_cells_le_8pct"] and not hi["pooled_hi_le_6.5pct"]
    ok["d3_layer2_over"] = d3_rule(cells, {(8, 16): (30, 600), (16, 8): (49, 600)}, (300, 500))["verdict"] == "fails"
    ok["d3_layer2_8pct_ok"] = d3_rule(cells, {(8, 16): (48, 600), (16, 8): (30, 600)}, (300, 500))["verdict"] == "works"

    # D4: build pairs with given marginals and discordance
    def pairs(both, p2_only, n3_only, n=500):
        a = np.r_[np.ones(both + p2_only, bool), np.zeros(n3_only, bool), np.zeros(n - both - p2_only - n3_only, bool)]
        b = np.r_[np.ones(both, bool), np.zeros(p2_only, bool), np.ones(n3_only, bool), np.zeros(n - both - p2_only - n3_only, bool)]
        return a, b
    v = lambda *x: d4_rule(*pairs(*x))["verdict"]  # noqa: E731
    ok["d4_separates"] = v(150, 100, 0) == "PBO separates edge from spread"
    ok["d4_not"] = v(200, 20, 0) == "PBO does not separate them" and v(200, 0, 24) == "PBO does not separate them"
    ok["d4_unclear_mid"] = v(200, 35, 0) == "unclear"
    ok["d4_unclear_n3"] = v(100, 0, 60) == "unclear (N3 - P2 >= 10 percentage points)"
    ok["d4_unclear_10pp_not_significant"] = d4_rule(*pairs(0, 4, 0, n=40))["verdict"] == "unclear"  # p = 0.125
    ok["d4_no_discordant"] = v(0, 0, 0) == "PBO does not separate them" and mcnemar_exact(0, 0) == 1.0
    ok["d4_exact_mcnemar"] = abs(mcnemar_exact(10, 2) - 2 * binom.cdf(2, 12, 0.5)) < 1e-15 and mcnemar_exact(5, 5) == 1.0
    # p-value pass rule in integers
    ok["p_pass"] = (list(p_pass([9, 10], [199, 199])) == [True, False]) and (list(p_pass([4, 5], [99, 99])) == [True, False])
    # outcome table, top to bottom
    ok["outcome"] = ([outcome_rule(*x)["row"] for x in (("design-invariant", "design-invariant", "works"),
                                                        ("intermediate", "design-dependent", "fails"),
                                                        ("design-dependent", "intermediate", "size only"),
                                                        ("design-invariant", "design-invariant", "size only"),
                                                        ("design-invariant", "intermediate", "fails"),
                                                        ("intermediate", "intermediate", "size only"))]
                     == [1, 2, 2, 3, 4, 4])
    ok["outcome_d2_mechanism"] = "as the mechanism" in outcome_rule("design-dependent", "x", "fails", "responds")["paper"] \
        and "without a mechanism" in outcome_rule("design-dependent", "x", "fails", "unclear")["paper"]
    print(json.dumps(ok, indent=1))
    print("SELFTEST (fabricated toy counts, not results):", "PASS" if all(ok.values()) else "FAIL")
    return all(ok.values())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    args = sys.argv[1:]
    dev_n = int(args[args.index("--dev-n") + 1]) if "--dev-n" in args else None
    pos = [x for i, x in enumerate(args) if not x.startswith("--") and (i == 0 or args[i - 1] != "--dev-n")]
    main(pos[0] if pos else OUT_DEFAULT, dev_n)

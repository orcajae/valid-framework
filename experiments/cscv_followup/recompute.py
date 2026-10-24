"""Standalone verifier for the follow-up to E17: PBO with its probability of loss, and CSCV with an embargo
(CHANGELOG E18). numpy, scipy and pandas only; it does not import run.py.

Reads results/cscv_followup/ (and E17's results/cscv_calibration/ for the canonical Layer 2 block statistics)
and recomputes, from the published files:
  - the sha256 of the row and block-statistics files against the run's records in checks/ and decision.json;
  - every decision of pre-registration section 4 (F1-F4) and every F5 count, from layer1_rows.csv.gz, with
    integer comparisons, and compares them with decision.json;
  - every count, n and mean in rates_layer1.csv and rates_layer2.csv, from the row files;
  - every Layer 2 canonical PBO and loss count (all sub-grids, S = 8 and 16; arm R excluded) from E17's
    stage3_blockstats.npz, and every embargoed one (e = 20) from layer2_blockstats_embargo.npz; and that each
    embargoed block holds 20 rows fewer than the canonical block;
  - with --regen N: the 1,000 reference-cell P2 and N3 series behind F1, plus N Layer 1 series drawn at random
    from all 216 cells and six arms, regenerated from their seeds with a generator written from the
    pre-registrations, and their four counts (k, loss, kE, lossE).
Not recomputed (it needs the unpublished per-event return matrices): arm R's 16 Layer 2 rows.
Prints the number of comparisons and of mismatches of each check, and exits 1 on any mismatch or on a check
that made no comparison.

usage: python3 -I experiments/cscv_followup/recompute.py [results/cscv_followup] [results/cscv_calibration]
                                                          [--regen N]
"""
import hashlib
import json
import math
import sys
from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.special import ndtr, stdtrit
from scipy.stats import binom

argv = sys.argv[1:]
REGEN = -1
if "--regen" in argv:
    i = argv.index("--regen")
    REGEN = int(argv[i + 1])
    del argv[i:i + 2]
D = argv[0] if len(argv) > 0 else "results/cscv_followup"
E17 = argv[1] if len(argv) > 1 else "results/cscv_calibration"
EPS = 1e-10
GRID = list(product([250, 1000, 2500], [4, 8, 16, 32], [8, 16], [0.0, 0.5, 0.9], ["IID", "GARCH", "MA10"]))
REF = GRID.index((1000, 8, 16, 0.5, "GARCH"))
REPS = {"N1": 1000, "N2": 1000, "N3": 500, "P1": 500, "P2": 500, "P3": 500}
COLS = {4: [0, 4, 8, 12], 8: [0, 2, 4, 6, 8, 10, 12, 14], 16: list(range(16)), 24: list(range(24))}
TH = (5, 10, 20, 50)                       # thresholds in hundredths: 0.05, 0.10, 0.20, 0.50
mismatch, compared, out = {}, {}, {}


def note(name, bad, n):
    mismatch[name], compared[name] = int(bad), int(n)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def below(k, K, t):                        # k / K < t / 100, exactly
    return 100 * k < t * K


# ------------------------------------------------------------------ hashes
dec = json.load(open(f"{D}/decision.json"))
d1 = json.load(open(f"{D}/checks/layer1_done.json"))
d2 = json.load(open(f"{D}/checks/layer2_done.json"))
h1, h2, hb = sha(f"{D}/layer1_rows.csv.gz"), sha(f"{D}/layer2_rows.csv"), sha(f"{D}/layer2_blockstats_embargo.npz")
note("sha256_records", (h1 != d1["rows_sha256"]) + (h1 != dec["layer1_rows_sha256"]) + (h2 != d2["rows_sha256"])
     + (h2 != dec["layer2_rows_sha256"]) + (hb != d2["blockstats_sha256"]), 5)

# ------------------------------------------------------------------ decisions from the Layer 1 rows
L = pd.read_csv(f"{D}/layer1_rows.csv.gz")
note("layer1_row_count", int(len(L) != 216 * sum(REPS.values())), 1)
for v, kc, lc in (("canonical", "k", "loss"), ("embargoed", "kE", "lossE")):
    for t in TH:
        L[f"P{t}_{v}"] = below(L[kc], L.K, t)
        L[f"L{t}_{v}"] = below(L[lc], L.K, t)
C = L["P20_canonical"] & L["L5_canonical"]
ref = L.cell == REF

# F1
p2 = L[ref & (L.arm == "P2")].sort_values("j")
n3 = L[ref & (L.arm == "N3")].sort_values("j")
a, b = C[p2.index].values, C[n3.index].values
kp, kn, bo, co = int(a.sum()), int(b.sum()), int((a & ~b).sum()), int((~a & b).sum())
diff = Fraction(100 * (kp - kn), 500)
pm = 1.0 if bo + co == 0 else min(1.0, 2 * float(binom.cdf(min(bo, co), bo + co, 0.5)))
f1 = "separates" if diff >= 10 and pm < 0.05 else ("does not separate" if abs(diff) < 5 else "unclear")
F1 = dec["F1"]
bad = sum([kp != F1["P2_pass"], kn != F1["N3_pass"], bo != F1["P2_only"], co != F1["N3_only"],
           float(diff) != F1["P2_minus_N3_pp"], abs(pm - F1["p_M"]) > 1e-12 * max(pm, 1e-300), f1 != F1["verdict"],
           p2.j.tolist() != list(range(500)), n3.j.tolist() != list(range(500))])
note("F1", bad, 9)
out["F1"] = {"P2": kp, "N3": kn, "P2_only": bo, "N3_only": co, "diff_pp": float(diff), "p_M": pm, "verdict": f1}


# F2: L(0.05) without embargo, IID and GARCH cells; MA10 beside it
def r95(n, pbar, cells):
    s = np.random.default_rng(0).binomial(n, pbar, size=(10_000, cells))
    sp = np.sort(s.max(1) - s.min(1))
    pos = Fraction(95 * 9_999, 100)
    i = int(pos)
    return int(sp[i]) + (pos - i) * int(sp[i + 1] - sp[i])


def level(col, serials):
    res = {}
    for arm in ("N1", "N2", "N3"):
        x = L[(L.arm == arm) & L.serial.isin(serials)]
        ks = x.groupby("cell")[col].sum().astype(int)
        n1 = REPS[arm]
        assert (x.groupby("cell").size() == n1).all()
        k, n = int(ks.sum()), n1 * len(ks)
        span = int(ks.max() - ks.min())
        R = r95(n1, k / n, len(ks))
        res[arm] = {"cells": len(ks), "k": k, "n": n, "wilson_hi": wilson(k, n)[1],
                    "span_pp": float(Fraction(100 * span, n1)), "R95_pp": float(R * Fraction(100, n1)),
                    "excess": Fraction(100, n1) * (span - R)}
    return res


f2a = level("L5_canonical", ("IID", "GARCH"))
f2 = "level holds" if all(r["wilson_hi"] <= 0.065 and r["excess"] <= 2 for r in f2a.values()) else "level fails"
bad = n = 0
for arm, r in f2a.items():
    s = dec["F2"]["arms"][arm]
    for key in ("cells", "k", "n", "span_pp", "R95_pp"):
        bad += r[key] != s[key]
        n += 1
    bad += abs(r["wilson_hi"] - s["wilson_hi"]) > 1e-12
    bad += float(r["excess"]) != s["span_minus_R95_pp"]
    n += 2
bad += f2 != dec["F2"]["verdict"]
for v in ("canonical", "embargoed"):
    for arm, r in level(f"L5_{v}", ("MA10",)).items():
        s = dec["F2"]["MA10_beside_no_verdict"][v][arm]
        for key in ("cells", "k", "n", "span_pp", "R95_pp"):
            bad += r[key] != s[key]
            n += 1
        bad += float(r["excess"]) != s["span_minus_R95_pp"]
        n += 1
note("F2", bad, n + 1)
out["F2"] = {"verdict": f2, **{a_: {"k": r["k"], "n": r["n"], "excess_pp": float(r["excess"])}
                                for a_, r in f2a.items()}}

# F3
cnt = {arm: (int(C[ref & (L.arm == arm)].sum()), int((ref & (L.arm == arm)).sum())) for arm in REPS}
pc = {arm: Fraction(100 * k, n) for arm, (k, n) in cnt.items()}
f3 = ("not useful" if any(pc[x] > 10 for x in ("N1", "N2", "N3")) else
      ("useful" if pc["P2"] >= 50 else "safe but weak"))
note("F3", sum(list(cnt[x]) != dec["F3"]["counts"][x] for x in REPS) + (f3 != dec["F3"]["verdict"]), 7)
out["F3"] = {"counts": cnt, "verdict": f3}

# F4: N1, P(0.20), T = 250, pooled over the 24 cells of each serial structure
k4 = {}
for ser, v in product(("MA10", "IID", "GARCH"), ("canonical", "embargoed")):
    x = L[(L.arm == "N1") & (L.serial == ser) & (L["T"] == 250)]
    assert len(x) == 24000
    k4[f"{ser}_{v}"] = int(x[f"P20_{v}"].sum())
ex0 = Fraction(100 * (k4["MA10_canonical"] - k4["IID_canonical"]), 24000)
exE = Fraction(100 * (k4["MA10_embargoed"] - k4["IID_embargoed"]), 24000)
f4 = ("premise not reproduced" if ex0 < 3 else "restores" if abs(exE) <= 1 else "overshoots" if exE < -1 else
      "does not restore" if exE >= 3 else "partial")
F4 = dec["F4"]
note("F4", sum(k4[x] != F4["N1_PBO020_counts_T250_of_24000"][x] for x in k4)
     + (float(ex0) != F4["excess0_pp"]) + (float(exE) != F4["excessE_pp"]) + (f4 != F4["verdict"]), len(k4) + 3)
out["F4"] = {"counts": k4, "excess0_pp": float(ex0), "excessE_pp": float(exE), "verdict": f4}

# F5: P2, by serial structure x T, and in the reference cell
bad = n = 0
groups = {f"{s}_T{T}": (L.arm == "P2") & (L.serial == s) & (L["T"] == T)
          for s, T in product(("IID", "GARCH", "MA10"), (250, 1000, 2500))}
groups["reference_cell"] = ref & (L.arm == "P2")
for g, m in groups.items():
    for v in ("canonical", "embargoed"):
        got = {f"P0.20_{v}": [int(L.loc[m, f"P20_{v}"].sum()), int(m.sum())],
               f"C_{v}": [int((L.loc[m, f"P20_{v}"] & L.loc[m, f"L5_{v}"]).sum()), int(m.sum())]}
        for key, val in got.items():
            bad += val != dec["F5"][g][key]
            n += 1
note("F5", bad, n)

# ------------------------------------------------------------------ rates files from the rows
r1 = pd.read_csv(f"{D}/rates_layer1.csv")
bad = n = 0
for v in ("canonical", "embargoed"):
    stats = {f"P{t / 100:.2f}": L[f"P{t}_{v}"] for t in TH}
    stats.update({f"L{t / 100:.2f}": L[f"L{t}_{v}"] for t in TH})
    stats.update({f"C{t / 100:.2f}_{u / 100:.2f}": L[f"P{t}_{v}"] & L[f"L{u}_{v}"] for t in TH for u in TH})
    kc, lc = ("k", "loss") if v == "canonical" else ("kE", "lossE")
    sub = r1[r1.variant == v].set_index(["cell", "arm", "statistic"])
    for name, m in stats.items():
        g = m.groupby([L.cell, L.arm]).agg(["sum", "count"])
        s = sub.xs(name, level="statistic")
        j = g.join(s[["k", "n"]], how="outer")
        bad += int(((j["sum"] != j["k"]) | (j["count"] != j["n"])).sum())
        n += len(j)
    for name, col in (("mean_PBO", kc), ("mean_PoL", lc)):
        mm = (L[col] / L.K).groupby([L.cell, L.arm]).mean()
        s = sub.xs(name, level="statistic")["rate"]
        j = pd.concat([mm, s], axis=1, join="outer")
        bad += int((~np.isclose(j.iloc[:, 0], j.iloc[:, 1], rtol=0, atol=1e-12)).sum())
        n += len(j)
bad += len(r1) != n
note("rates_layer1", bad, n)

L2 = pd.read_csv(f"{D}/layer2_rows.csv", dtype={"seed": str})
r2 = pd.read_csv(f"{D}/rates_layer2.csv").set_index(["arm", "M", "S", "e", "statistic"])
bad = n = 0
for (arm, M, S, e), g in L2.groupby(["arm", "M", "S", "e"]):
    st = {f"P{t / 100:.2f}": below(g.k, g.K, t) for t in TH}
    st.update({f"L{t / 100:.2f}": below(g.loss, g.K, t) for t in TH})
    st.update({f"C{t / 100:.2f}_{u / 100:.2f}": below(g.k, g.K, t) & below(g.loss, g.K, u) for t in TH for u in TH})
    for name, m in st.items():
        s = r2.loc[(arm, M, S, e, name)]
        bad += (int(m.sum()) != s["k"]) or (len(m) != s["n"])
        n += 1
    for name, col in (("mean_PBO", "k"), ("mean_PoL", "loss")):
        bad += abs(float((g[col] / g.K).mean()) - r2.loc[(arm, M, S, e, name), "rate"]) > 1e-12
        n += 1
bad += len(r2) != n
note("rates_layer2", bad, n)

# ------------------------------------------------------------------ Layer 2 counts from block statistics
HALVES = {}


def counts(nb, s1, s2, S):
    if S not in HALVES:
        cs = list(combinations(range(S), S // 2))
        H = np.zeros((len(cs), S))
        for i_, c_ in enumerate(cs):
            H[i_, list(c_)] = 1.0
        HALVES[S] = H
    I = HALVES[S]

    def sr(J):
        nn = (J @ nb)[:, None]
        mu = (J @ s1) / nn
        return mu / (np.sqrt(np.maximum((J @ s2) / nn - mu * mu, 0.0)) + EPS)

    isr, oos = sr(I), sr(1 - I)
    best = isr.argmax(1)
    ob = oos[np.arange(len(oos)), best]
    rank = (oos < ob[:, None]).sum(1) + ((oos == ob[:, None]).sum(1) + 1) / 2
    w = rank / (isr.shape[1] + 1)
    return len(oos), int((np.log(w / (1 - w)) <= 0).sum()), int((ob < 0).sum())


bc = np.load(f"{E17}/stage3_blockstats.npz")
be = np.load(f"{D}/layer2_blockstats_embargo.npz")
idx = L2.set_index(["arm", "seed", "M", "S", "e"])
bad = n = nb_bad = 0
for (arm, seed), _ in L2[L2.arm != "R"].groupby(["arm", "seed"]):
    for S in (8, 16):
        p0, pE = f"{arm}_{seed}_S{S}", f"{arm}_{seed}_S{S}_e20"
        nb_bad += int(not np.array_equal(be[pE + "_n"], bc[p0 + "_n"] - 20))
        for e, (pre, src) in ((0, (p0, bc)), (20, (pE, be))):
            nb, s1, s2 = src[pre + "_n"].astype(float), src[pre + "_sum"], src[pre + "_sumsq"]
            for M, cols in COLS.items():
                r = idx.loc[(arm, seed, M, S, e)]
                got = counts(nb, s1[:, cols], s2[:, cols], S)
                bad += got != (r.K, r.k, r.loss)
                n += 1
note("layer2_counts_from_blocks", bad, n)
note("layer2_embargo_block_sizes", nb_bad, n // 8)
note("layer2_rows_cover_all_but_R", int((L2.arm != "R").sum() != n), 1)

# ------------------------------------------------------------------ optional: regenerate Layer 1 series
if REGEN >= 0:
    ARM = {"N1": 1, "N2": 2, "N3": 3, "P1": 4, "P2": 5, "P3": 6}
    EDGE = {"P1": 0.5, "P2": 1.0, "P3": 2.0, "N3": 1.0}

    def gauss(rng, n_, M, rho):
        Z = rng.standard_normal((n_, M + 1))
        return math.sqrt(rho) * Z[:, :1] + math.sqrt(1 - rho) * Z[:, 1:]

    def shocks(rng, T, M, rho, ser):
        if ser == "IID":
            return gauss(rng, T, M, rho)
        if ser == "MA10":
            g = gauss(rng, T + 9, M, rho)
            e_ = np.zeros((T, M))
            for k_ in range(10):
                e_ = e_ + g[9 - k_:9 - k_ + T]
            return e_ / math.sqrt(10)
        eta = rng.standard_t(5, T + 500) / math.sqrt(5 / 3)
        g = gauss(rng, T, M, rho)
        eps = -np.sign(g) * stdtrit(5, ndtr(-np.abs(g))) / math.sqrt(5 / 3)
        h = [1.0]
        for t in range(1, T + 500):
            u = math.sqrt(h[t - 1]) * float(eta[t - 1])
            h.append(0.02 + 0.08 * u * u + 0.90 * h[t - 1])
        return np.sqrt(np.array(h[500:]))[:, None] * eps

    def series(cell, arm, j):
        T, M, S, rho, ser = GRID[cell]
        sr_ = np.zeros(M)
        if arm == "N2":
            sr_ = -1.0 * np.arange(M) / (M - 1)
        elif arm in EDGE:
            sr_[0] = EDGE[arm]
        mu = sr_ / math.sqrt(252) * 0.01
        if arm == "N3":
            mu = mu - mu[0]
        seed = [18, 5, cell, j] if arm == "N3" else [18, ARM[arm], cell, j]
        return mu + 0.01 * shocks(np.random.default_rng(seed), T, M, rho, ser), S

    def row_counts(R, S, e):
        blocks = [x[e:] for x in np.array_split(np.arange(R.shape[0]), S)]
        nb = np.array([len(x) for x in blocks], float)
        return counts(nb, np.stack([R[x].sum(0) for x in blocks]), np.stack([(R[x] ** 2).sum(0) for x in blocks]), S)

    pick = pd.concat([L[ref & L.arm.isin(["P2", "N3"])],
                      L[~(ref & L.arm.isin(["P2", "N3"]))].sample(REGEN, random_state=20261025)])
    bad = 0
    for _, rr in pick.iterrows():
        R, S = series(int(rr.cell), rr.arm, int(rr.j))
        K0, k0, l0 = row_counts(R, S, 0)
        _, kE, lE = row_counts(R, S, 9)
        bad += (K0, k0, l0, kE, lE) != (rr.K, rr.k, rr.loss, rr.kE, rr.lossE)
    note("layer1_regenerated", bad, len(pick))

empty = [k for k, v in compared.items() if v == 0]
print(json.dumps({"decisions": out, "mismatches": mismatch, "comparisons": compared,
                  "checks_with_no_comparison": empty}, indent=1, default=str))
sys.exit(1 if any(mismatch.values()) or empty else 0)

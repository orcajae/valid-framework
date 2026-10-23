"""Standalone verifier for the PBO-threshold calibration study (CHANGELOG E17). numpy, scipy and pandas only;
it does not import run.py or summarize.py.

Reads results/cscv_calibration/ and recomputes, from the published files:
  - every decision of pre-registration section 4 (D1-D4) and the "Use of the outcomes" row, from the rows,
    and compares them with decision.json;
  - Stage 2's observed PBO count against Stage 1's, series by series;
  - every calibrated-p count (resamples with PBO* <= observed PBO) from the stored resample counts, Stages 2-3;
  - every Layer 2 PBO count (all sub-grids, S = 8 and 16; arm R excluded) from the published block statistics;
  - with --regen N: N Layer 1 grid series (cells 0-215) drawn at random, regenerated from their seeds with a
    generator written from the pre-registration and the implementation notes of run.py, and their PBO counts.
    N = 2000 draws the same series as the check reported in CHANGELOG E17.
Not recomputed (they need the unpublished per-event return matrices): the calibrated p-values from their
bootstrap streams, and arm R's PBO counts.
Prints the number of comparisons and of mismatches of each check, and exits 1 on any mismatch or on a check
that made no comparison.

usage: python3 -I experiments/cscv_calibration/recompute.py [results/cscv_calibration] [--regen N]
"""
import json
import math
import sys
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.special import ndtr, stdtrit
from scipy.stats import binom, fisher_exact

argv = sys.argv[1:]
REGEN = 0
if "--regen" in argv:
    i = argv.index("--regen")
    REGEN = int(argv[i + 1])
    del argv[i:i + 2]
D = argv[0] if argv else "results/cscv_calibration"
EPS = 1e-10
GRID = list(product([250, 1000, 2500], [4, 8, 16, 32], [8, 16], [0.0, 0.5, 0.9], ["IID", "GARCH", "MA10"]))
REF = GRID.index((1000, 8, 16, 0.5, "GARCH"))
S2CELLS = sorted({GRID.index((T, M, 8, 0.5, s)) for T in (250, 1000) for M in (4, 8, 16)
                  for s in ("IID", "GARCH", "MA10")} | {REF})
COLS = {4: [0, 4, 8, 12], 8: [0, 2, 4, 6, 8, 10, 12, 14], 16: list(range(16)), 24: list(range(24))}
mismatch = {}
compared = {}


def note(name, n, k=None):
    mismatch[name] = int(n)
    if k is not None:
        compared[name] = int(k)


def passes_p(le, B):
    return 20 * (1 + le) <= B + 1          # p = (1 + le) / (B + 1) <= 0.05


def wilson_hi(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n)) / d + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def mcnemar(a, b):
    b_, c_ = int((a & ~b).sum()), int((~a & b).sum())
    return b_, c_, (1.0 if b_ + c_ == 0 else min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5)))


# ------------------------------------------------------------------ decisions from the rows
r1 = pd.read_csv(f"{D}/stage1_rows.csv.gz")
r2 = pd.read_csv(f"{D}/stage2_rows.csv")
r3 = pd.read_csv(f"{D}/stage3_rows.csv")
dec = json.load(open(f"{D}/decision.json"))
L1 = r1[r1.cell < 216]
pbo_lt = lambda df, t: (df.k_first / df.K) < t  # noqa: E731

d1 = {}
n1 = L1[L1.arm == "N1"]
for tau, key in ((0.20, "tau=0.20"), (0.05, "tau=0.05")):
    rate = pbo_lt(n1, tau).groupby(n1.cell).mean()
    pbar = pbo_lt(n1, tau).mean()
    draws = np.random.default_rng(0).binomial(1000, pbar, size=(10000, 216)) / 1000
    r95 = np.percentile(draws.max(1) - draws.min(1), 95)
    ex = round((rate.max() - rate.min() - r95) * 100, 1)
    d1[key] = "design-dependent" if ex > 5 else ("design-invariant" if ex <= 2 else "intermediate")
note("D1", sum(d1[k] != dec["D1"][k]["verdict"] for k in d1))

ref = L1[L1.cell == REF]
k1, k2 = int(pbo_lt(ref[ref.arm == "N1"], 0.2).sum()), int(pbo_lt(ref[ref.arm == "N2"], 0.2).sum())
pF = fisher_exact([[k2, 1000 - k2], [k1, 1000 - k1]], alternative="greater")[1]
diff = (k2 - k1) / 10
d2 = "responds" if (diff >= 10 and pF < 0.05) else ("does not respond" if diff < 3 else "unclear")
note("D2", (d2 != dec["D2"]["verdict"]) + ([k1, k2] != [dec["D2"]["N1"][0], dec["D2"]["N2"][0]]))

r2["pass"] = passes_p(r2.le_L20, r2.B)
n1s = r2[r2.arm == "N1"]
per = n1s.groupby("cell")["pass"].mean()
A = r3[r3.arm.isin(["A-BTC", "A-ETH", "A-SOL"])]
l2 = {t: int(((A.valid == 1) & passes_p(A[f"cal_{t}_le_L20"], A.B)).sum()) for t in ("M8_S16", "M16_S8")}
size_ok = (per <= 0.08).all() and wilson_hi(int(n1s["pass"].sum()), len(n1s)) <= 0.065 and \
    all(k / 600 <= 0.08 for k in l2.values())
p2k = int(r2[(r2.cell == REF) & (r2.arm == "P2")]["pass"].sum())
d3 = "works" if (size_ok and p2k / 500 >= 0.5) else ("size only" if size_ok else "fails")
note("D3", (d3 != dec["D3"]["verdict"]) + (l2["M8_S16"] != dec["D3"]["layer2_A"]["M8_S16"]["k"]))

P2 = ref[ref.arm == "P2"].sort_values("j")
N3 = ref[ref.arm == "N3"].sort_values("j")
assert P2.j.tolist() == N3.j.tolist() == list(range(500))
a, b = pbo_lt(P2, 0.2).values, pbo_lt(N3, 0.2).values
bb, cc, pq = mcnemar(a, b)
dd = (a.mean() - b.mean()) * 100
d4 = "PBO separates edge from spread" if (dd >= 10 and pq < 0.05) else \
    ("PBO does not separate them" if abs(dd) < 5 else "unclear")
note("D4", (d4 != dec["D4"]["PBO<0.20"]["verdict"]) + ((bb, cc) != (dec["D4"]["PBO<0.20"]["P2_only"],
                                                                     dec["D4"]["PBO<0.20"]["N3_only"])))
row = 1 if d3 == "works" else (2 if "design-dependent" in d1.values() else
                               (3 if set(d1.values()) == {"design-invariant"} else 4))
note("outcome_row", row != dec["use_of_outcomes"]["row"])

# ------------------------------------------------------------------ Stage 2 observed PBO against Stage 1
m = r2.merge(r1[["cell", "arm", "j", "k_first", "K"]], on=["cell", "arm", "j"], how="left")
note("stage2_k_obs_vs_stage1", (m.k_first.isna() | (m.k_obs != m.k_first) | (m.K_x != m.K_y)).sum(), len(m))

# ------------------------------------------------------------------ calibrated-p counts from resample counts
s2 = np.load(f"{D}/stage2_star.npz")
bad = n_cmp = 0
for (cell, arm), g in r2.groupby(["cell", "arm"]):
    g = g.sort_values("j")
    for L in ("L20", "L5", "L50"):
        key = f"c{cell:03d}_{arm}_pbo_star_counts_{L}"
        if g[f"le_{L}"].isna().all():
            continue                       # this (cell, arm) was not run at block length L
        if key not in s2.files:            # a row implies this key: a missing key is a mismatch
            bad += len(g)
            continue
        star = s2[key]
        le = (star <= g.k_obs.values[:, None]).sum(1)
        bad += int((le != g[f"le_{L}"].values).sum())
        n_cmp += len(g)
note("stage2_le_from_star", bad, n_cmp)

s3 = np.load(f"{D}/stage3_star.npz")
bad = n_cmp = 0
for _, rr in r3.iterrows():
    for tag, Ls in (("M8_S16", ("L20", "L5", "L50")), ("M16_S8", ("L20",))):
        for L in Ls:
            key = f"{rr.arm}_{int(rr.seed)}_{tag}_{L}_star"
            if pd.isna(rr[f"cal_{tag}_le_{L}"]):
                continue                   # not run for this series at block length L
            if key not in s3.files:        # a row implies this key: a missing key is a mismatch
                bad += 1
                continue
            bad += int((s3[key] <= rr[f"cal_{tag}_k_obs"]).sum() != rr[f"cal_{tag}_le_{L}"])
            n_cmp += 1
note("stage3_le_from_star", bad, n_cmp)


# ------------------------------------------------------------------ Layer 2 PBO from block statistics
HALVES = {}


def k_from_blocks(n, s1, s2_, S):
    if S not in HALVES:
        cs = list(combinations(range(S), S // 2))
        H = np.zeros((len(cs), S))
        for i, c in enumerate(cs):
            H[i, list(c)] = 1.0
        HALVES[S] = H
    I = HALVES[S]

    def sr(J):
        nn = (J @ n)[:, None]
        mu = (J @ s1) / nn
        return mu / (np.sqrt(np.maximum((J @ s2_) / nn - mu * mu, 0)) + EPS)

    isr, oos = sr(I), sr(1 - I)
    best = isr.argmax(1)
    ob = oos[np.arange(len(oos)), best][:, None]
    rank = (oos < ob).sum(1) + ((oos == ob).sum(1) + 1) / 2
    w = rank / (isr.shape[1] + 1)
    return int((np.log(w / (1 - w)) <= 0).sum())


bs = np.load(f"{D}/stage3_blockstats.npz")
bad = n_cmp = 0
for _, rr in r3[r3.arm != "R"].iterrows():
    for S in (8, 16):
        p = f"{rr.arm}_{int(rr.seed)}_S{S}"
        n, s1, sq = bs[p + "_n"].astype(float), bs[p + "_sum"], bs[p + "_sumsq"]
        for M, cols in COLS.items():
            n_cmp += 1
            bad += k_from_blocks(n, s1[:, cols], sq[:, cols], S) != rr[f"M{M}_S{S}_k_first"]
note("stage3_pbo_from_blocks", bad, n_cmp)

# ------------------------------------------------------------------ optional: regenerate Layer 1 series
if REGEN:
    ARM = {"N1": 1, "N2": 2, "N3": 3, "P1": 4, "P2": 5, "P3": 6}

    def gauss(rng, n, M, rho):
        Z = rng.standard_normal((n, M + 1))
        return math.sqrt(rho) * Z[:, :1] + math.sqrt(1 - rho) * Z[:, 1:]

    def shocks(rng, T, M, rho, ser):
        if ser == "IID":
            return gauss(rng, T, M, rho)
        if ser == "MA10":
            g = gauss(rng, T + 9, M, rho)
            e = np.zeros((T, M))
            for k in range(10):
                e = e + g[9 - k:9 - k + T]
            return e / math.sqrt(10)
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
        sig = np.full(M, 0.01)
        sr = np.zeros(M)
        if arm == "N2":
            sr = -1.0 * np.arange(M) / (M - 1)
        elif arm in ("P1", "P2", "P3", "N3"):
            sr[0] = {"P1": 0.5, "P2": 1.0, "P3": 2.0, "N3": 1.0}[arm]
        mu = sr / math.sqrt(252) * sig
        if arm == "N3":
            mu = mu - mu[0]
        e = shocks(np.random.default_rng([5, cell, j] if arm == "N3" else [ARM[arm], cell, j]), T, M, rho, ser)
        return mu + sig * e, S

    bad = 0
    for _, rr in L1.sample(REGEN, random_state=20261023).iterrows():  # grid cells only (0-215)
        R, S = series(int(rr.cell), rr.arm, int(rr.j))
        blocks = np.array_split(np.arange(R.shape[0]), S)
        n = np.array([len(x) for x in blocks], float)
        bad += k_from_blocks(n, np.stack([R[x].sum(0) for x in blocks]),
                             np.stack([(R[x] ** 2).sum(0) for x in blocks]), S) != rr.k_first
    note("layer1_regenerated", bad, REGEN)

empty = [k for k, v in compared.items() if v == 0]
print(json.dumps({"D1": d1, "D2": d2, "D3": d3, "D4": d4, "outcome_row": row, "mismatches": mismatch,
                  "comparisons": compared, "checks_with_no_comparison": empty}, indent=1))
sys.exit(1 if any(mismatch.values()) or empty else 0)

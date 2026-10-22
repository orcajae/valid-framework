"""Standalone verifier for the robustness study of the E14 findings (CHANGELOG E15). numpy and pandas only.

Reads results/inbox7/ and recomputes, with the conventions of experiments/inbox7/run.py:
  - S1, S2, S4 PBO by Eq. 1 from the stored *_is / *_oos matrices (per path: OOS rank of the
    IS-best configuration, ties averaged, IS-best by np.argmax, omega_bar = rank/(M+1),
    pass if logit(omega_bar) <= 0; paths with fewer than 2 valid configurations are skipped);
  - S3 by canonical CSCV from the stored block statistics *_S3_n, *_S3_sum, *_S3_sumsq
    (all C(16, 8) halves; Sharpe = mean / (sqrt(E[x^2] - mean^2) + 1e-10));
  - every T1-T4 verdict from the stored permuted AUCs (pass if the real holdout AUC >
    np.percentile(perm, 95); T1 on the holdout without purge, T2-T4 on the purged holdout).
Prints the number of mismatches against stage1_rows.csv and stage2_rows.csv.

usage: python3 -I experiments/inbox7/recompute.py results/inbox7
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

EPS = 1e-10
S = 16


def avg_rank(x):
    """Ranks 1..n with ties averaged (scipy.stats.rankdata, method='average')."""
    return pd.Series(x).rank(method="average").to_numpy()


def pbo_eq1(is_m, oos_m):
    cnt = tot = 0
    for j in range(is_m.shape[1]):
        a, b = is_m[:, j], oos_m[:, j]
        ok = ~(np.isnan(a) | np.isnan(b))
        M = int(ok.sum())
        if M < 2:
            continue
        best = int(np.argmax(a[ok]))
        wbar = avg_rank(b[ok])[best] / (M + 1)
        cnt += int(np.log(wbar / (1 - wbar)) <= 0)
        tot += 1
    return (cnt / tot if tot else np.nan), tot


HALVES = np.zeros((len(list(combinations(range(S), S // 2))), S))
for i, c in enumerate(combinations(range(S), S // 2)):
    HALVES[i, list(c)] = 1.0


def cscv_pbo(nb, s1, s2):
    # Sum each half directly; do not form the out-of-sample half as total minus in-sample:
    # that breaks exact ties between configurations with identical predictions on a half.
    nb = nb.astype(float)

    def sr(I):
        n = (I @ nb)[:, None]
        mu = (I @ s1) / n
        var = np.maximum((I @ s2) / n - mu ** 2, 0.0)
        return mu / (np.sqrt(var) + EPS)

    is_m, oos_m = sr(HALVES).T, sr(1.0 - HALVES).T  # M x C(16, 8)
    M = is_m.shape[0]
    best = is_m.argmax(0)
    ranks = pd.DataFrame(oos_m).rank(axis=0, method="average").to_numpy()
    w = ranks[best, np.arange(is_m.shape[1])] / (M + 1)
    return float(np.mean(np.log(w / (1 - w)) <= 0)), is_m.shape[1]


TOL = 1e-12  # stage1_rows.csv stores PBO with 15-16 significant digits (one CSV round trip),
# so a value can differ from the recomputed one in the last binary digit; larger differences count.


def same(a, b):
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= TOL


def main(d):
    r1 = pd.read_csv(os.path.join(d, "stage1_rows.csv"), float_precision="round_trip")
    r2 = pd.read_csv(os.path.join(d, "stage2_rows.csv"), float_precision="round_trip")
    z1 = np.load(os.path.join(d, "stage1_arrays.npz"))
    z2 = np.load(os.path.join(d, "stage2_arrays.npz"))

    n_pbo = mis_pbo = mis_paths = n_ulp = 0
    max_diff = 0.0
    for _, row in r1.iterrows():
        k = f"{row.arm}_{row.asset}_{int(row.seed)}"
        for tag in ("S1", "S2", "S4"):
            p, n = pbo_eq1(z1[f"{k}_{tag}_is"], z1[f"{k}_{tag}_oos"])
            n_pbo += 1
            mis_pbo += not same(p, row[f"pbo_{tag}"])
            n_ulp += 0 < abs(p - row[f"pbo_{tag}"]) <= TOL
            mis_paths += n != row[f"paths_{tag}"]
            max_diff = max(max_diff, abs(p - row[f"pbo_{tag}"]))
        p, n = cscv_pbo(z1[f"{k}_S3_n"], z1[f"{k}_S3_sum"], z1[f"{k}_S3_sumsq"])
        n_pbo += 1
        mis_pbo += not same(p, row["pbo_S3"])
        n_ulp += 0 < abs(p - row["pbo_S3"]) <= TOL
        mis_paths += n != row["paths_S3"]
        max_diff = max(max_diff, abs(p - row["pbo_S3"]))

    n_t = mis_t = mis_p95 = 0
    for _, row in r2.iterrows():
        k = f"{row.arm}_{row.asset}_{int(row.seed)}"
        for tag, h in (("T1", "u"), ("T2", "p"), ("T3", "p"), ("T4", "p")):
            p95 = float(np.percentile(z2[f"{k}_{tag}_aucs"], 95))
            verdict = bool(row[f"auc_{h}"] > p95)
            n_t += 1
            mis_t += verdict != bool(row[f"{tag}_pass"])
            mis_p95 += p95 != row[f"{tag}_p95"]

    print(f"PBO: {mis_pbo} mismatches of {n_pbo} values ({len(r1)} series x S1-S4); "
          f"path-count mismatches {mis_paths}; max |diff| {max_diff:.3g} "
          f"({n_ulp} values differ only below {TOL:g}, from the CSV round trip)")
    print(f"Permutation: {mis_t} verdict mismatches of {n_t} ({len(r2)} series x T1-T4); "
          f"95th-percentile mismatches {mis_p95}")
    return 0 if (mis_pbo == mis_t == mis_paths == mis_p95 == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/inbox7"))

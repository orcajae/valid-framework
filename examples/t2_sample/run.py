"""VALID Tier 2 sample review: gate size and power for a fictional client ("Sample Fund A"), synthetic data only.

This is a demonstration of the deliverable of a Tier 2 review, not a study. No market data is read: every series
comes from the fixed-parameter generator below. Design and pass rules: PREREGISTRATION.md (same folder).

usage (python3 -I -B, at most 2 workers):
  run.py selftest                    decision rules on fabricated counts
  run.py smoke --out OUT             timing only, 5 series per arm; no pass rate is printed or stored
  run.py run --out OUT [--workers 2] the registered run (refuses unless both files are committed and unchanged)
  run.py summarize --out OUT         decision.json from rows.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from itertools import combinations, product  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION.md")

# ---------------------------------------------------------------- design (PREREGISTRATION.md section 2)
T, BURN = 1000, 250                      # trading days kept, burn-in days dropped
SIGMA, ALPHA, BETA = 0.02, 0.08, 0.90    # GARCH(1,1): unconditional daily vol, ARCH, GARCH
OMEGA = SIGMA ** 2 * (1 - ALPHA - BETA)
NU = 5                                   # Student-t innovations, standardised to unit variance
PHI = 0.9                                # AR(1) persistence of the flow feature X
ARMS = {                                 # name: (daily drift mu, planted loading b on X)
    "N0": (0.0005, 0.0),
    "N1": (0.0, 0.0),
    "P1": (0.0, 0.04),
    "P2": (0.0, 0.08),
}
REPS = 200
HORIZON = 5                              # label: sign of the next 5 daily returns
WINDOWS = (5, 20, 60)                    # trailing-return features
TRAIN0, STEP = 400, 100                  # walk-forward: first fit on 400 usable days, refit every 100
PURGE = HORIZON                          # training rows whose labels reach into the test block are dropped
FEATURE_SETS = {"A": ("r5", "r20", "x"), "B": ("r20", "r60", "x")}
CS = (0.1, 1.0)                          # L2 strength of the logistic regression (sklearn's C convention)
BANDS = (0.0, 0.02)                      # no position while |p - 0.5| <= band
CONFIGS = list(product(FEATURE_SETS, CS, BANDS))   # 8 configurations, last factor fastest
COST = 0.0009                            # 9 bp per unit change in position: 18 bp for a round trip in and out
COSTS_DESC = (0.0, 0.00045, 0.0009, 0.0018)        # descriptive cost sensitivity
S_CSCV = 16
N_PERM = 100
SEED = (2026, 1024)
EPS = 1e-10
SIZE_MAX, POWER_MIN = 15, 100            # of 200: size within bound if N1 passes <= 15; adequate power if P2 >= 100
GATES = ("G1", "G2", "G3", "G4", "Gall")


def need(cond, msg):
    if not cond:
        raise SystemExit("STOP: " + msg)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- generator
def shocks(rep):
    """Shared draws of one replication: return innovations z and flow innovations e (all arms use the same)."""
    rng = np.random.default_rng(list(SEED) + [rep])
    z = rng.standard_t(NU, size=BURN + T) / math.sqrt(NU / (NU - 2))
    e = rng.standard_normal(BURN + T)
    x0 = rng.standard_normal()
    return z, e, x0


def series(arm, rep):
    """Daily returns r (T,) and the flow feature x (T,) of one series."""
    mu, b = ARMS[arm]
    z, e, x0 = shocks(rep)
    n = BURN + T
    x = np.empty(n)
    x[0] = x0
    c = math.sqrt(1 - PHI ** 2)
    for t in range(1, n):
        x[t] = PHI * x[t - 1] + c * e[t]
    r = np.empty(n)
    h = SIGMA ** 2
    r[0] = mu + math.sqrt(h) * z[0]
    s = math.sqrt(1 - b * b)
    for t in range(1, n):
        eps_prev = r[t - 1] - mu
        h = OMEGA + ALPHA * eps_prev * eps_prev + BETA * h
        r[t] = mu + math.sqrt(h) * (b * x[t - 1] + s * z[t])
    return r[BURN:], x[BURN:]


def design(r, x):
    """Features at the close of day t, the label of day t, and the usable days."""
    csum = np.concatenate([[0.0], np.cumsum(r)])
    F = {"x": x}
    for w in WINDOWS:
        v = np.full(T, np.nan)
        v[w - 1:] = (csum[w:] - csum[:-w]) / (SIGMA * math.sqrt(w))
        F[f"r{w}"] = v
    fwd = np.full(T, np.nan)
    fwd[:T - HORIZON] = csum[1 + HORIZON:] - csum[1:T - HORIZON + 1]
    y = (fwd > 0).astype(float)
    usable = np.arange(max(WINDOWS) - 1, T - HORIZON)
    return F, y, usable


# ---------------------------------------------------------------- model
def fit_logit(Xtr, ytr, C):
    """L2 logistic regression by Newton steps: minimise C * sum(logloss) + 0.5 * |w|^2 (intercept unpenalised)."""
    n, d = Xtr.shape
    A = np.hstack([np.ones((n, 1)), Xtr])
    w = np.zeros(d + 1)
    P = np.eye(d + 1) / C
    P[0, 0] = 0.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-A @ w))
        g = A.T @ (p - ytr) + P @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + P + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def walk_forward(F, y, usable, cfg):
    """Out-of-sample probabilities on the test days of an expanding walk-forward with a purge."""
    fs, C, _ = cfg
    X = np.column_stack([F[k] for k in FEATURE_SETS[fs]])
    oos = usable[TRAIN0:]
    p = np.empty(len(oos))
    for s in range(0, len(oos), STEP):
        start = oos[s]
        tr = usable[usable < start - PURGE + 1]
        mu, sd = X[tr].mean(0), X[tr].std(0) + EPS
        w = fit_logit((X[tr] - mu) / sd, y[tr], C)
        te = oos[s:s + STEP]
        p[s:s + len(te)] = 1.0 / (1.0 + np.exp(-(w[0] + ((X[te] - mu) / sd) @ w[1:])))
    return oos, p


def pnl(r, oos, p, band, cost):
    pos = np.where(p > 0.5 + band, 1.0, np.where(p < 0.5 - band, -1.0, 0.0))
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * r[oos + 1] - cost * turn


def sharpe(v):
    return float(v.mean() / (v.std() + EPS) * math.sqrt(252))


# ---------------------------------------------------------------- statistics
_HALVES = {}


def halves(S):
    if S not in _HALVES:
        cs = list(combinations(range(S), S // 2))
        H = np.zeros((len(cs), S))
        for i, c in enumerate(cs):
            H[i, list(c)] = 1.0
        _HALVES[S] = H
    return _HALVES[S]


def cscv_pbo(R, S):
    """Canonical CSCV as in valid-framework experiments/cscv_calibration (E17): returns (K, k)."""
    blocks = np.array_split(np.arange(R.shape[0]), S)
    n = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([R[b].sum(0) for b in blocks])
    s2 = np.stack([(R[b] ** 2).sum(0) for b in blocks])
    ind = halves(S)

    def sr(J):
        nn = (J @ n)[:, None]
        m = (J @ s1) / nn
        return m / (np.sqrt(np.maximum((J @ s2) / nn - m * m, 0.0)) + EPS)

    isr, osr = sr(ind), sr(1.0 - ind)
    K, M = isr.shape
    best = isr.argmax(1)
    ob = osr[np.arange(K), best]
    rank = (osr < ob[:, None]).sum(1) + ((osr == ob[:, None]).sum(1) + 1) / 2
    wv = rank / (M + 1)
    return K, int(np.sum(np.log(wv / (1 - wv)) <= 0))


def auc_parts(p, y):
    """AUC as an exact fraction: (2 * U, 2 * n1 * n0), with average ranks for ties."""
    r2 = (2 * rankdata(p)).astype(np.int64)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    u2 = int(r2[pos].sum()) - n1 * (n1 + 1)
    return u2, 2 * n1 * n0


def auc(p, y):
    a, b = auc_parts(p, y)
    return a / b


def perm_count(p, y, labels):
    """Number of relabelled AUCs at or above the observed one (each row of labels is one relabelling)."""
    a0, _ = auc_parts(p, y)
    r2 = (2 * rankdata(p)).astype(np.int64)
    n1 = int(y.sum())
    u2 = (labels * r2[None, :]).sum(1).astype(np.int64) - n1 * (n1 + 1)
    return int(np.sum(u2 >= a0))


def evaluate(args):
    arm, rep = args
    r, x = series(arm, rep)
    F, y, usable = design(r, x)
    probs, oos = [], None
    for cfg in CONFIGS:
        oos, p = walk_forward(F, y, usable, cfg)
        probs.append(p)
    yo = y[oos]
    out = {"arm": arm, "rep": rep, "n_oos": len(oos)}
    by_cost = {}
    for c in COSTS_DESC:
        R = np.column_stack([pnl(r, oos, probs[i], CONFIGS[i][2], c) for i in range(len(CONFIGS))])
        sh = [sharpe(R[:, i]) for i in range(R.shape[1])]
        sel = int(np.argmax(sh))
        by_cost[c] = (R, sh, sel)
    R, sh, sel = by_cost[COST]
    K, k = cscv_pbo(R, S_CSCV)
    p_sel = probs[sel]
    a2, b2 = auc_parts(p_sel, yo)
    rng = np.random.default_rng([SEED[0], 2048, list(ARMS).index(arm), rep])   # relabelling stream
    n = len(yo)
    lab_iid = np.stack([rng.permutation(yo) for _ in range(N_PERM)])
    offs = rng.integers(20, n - 20, size=N_PERM)
    lab_circ = np.stack([np.roll(yo, int(o)) for o in offs])
    k_iid = perm_count(p_sel, yo, lab_iid)
    k_circ = perm_count(p_sel, yo, lab_circ)
    g1 = 5 * k < K                                   # PBO < 0.20
    g2 = Fraction(1 + k_iid, N_PERM + 1) < Fraction(1, 20)
    g3 = 20 * a2 > 11 * b2                           # AUC > 0.55
    g4 = R[:, sel].mean() > 0                        # net Sharpe ratio > 0
    out.update({
        "sel": sel, "sel_config": "%s|C=%s|band=%s" % CONFIGS[sel], "sharpe_sel": round(sh[sel], 6),
        "auc_sel": round(a2 / b2, 6), "auc_u2": a2, "auc_den": b2,
        "pbo_K": K, "pbo_k": k, "perm_k_iid": k_iid, "perm_k_circ": k_circ,
        "G1": int(g1), "G2": int(g2), "G3": int(g3), "G4": int(g4), "Gall": int(g1 and g2 and g3 and g4),
        "G2c": int(Fraction(1 + k_circ, N_PERM + 1) < Fraction(1, 20)),
    })
    for c, (Rc, shc, selc) in by_cost.items():
        out["G4_cost_%gbp" % round(c * 2e4, 1)] = int(Rc[:, selc].mean() > 0)
    return out


# ---------------------------------------------------------------- decisions (exact, integer)
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def size_verdict(k):
    return "size within bound" if k <= SIZE_MAX else "size exceeds bound"


def power_verdict(k):
    return "adequate power" if k >= POWER_MIN else "weak power"


def selftest(quiet=False):
    t = [size_verdict(15) == "size within bound", size_verdict(16) == "size exceeds bound",
         power_verdict(99) == "weak power", power_verdict(100) == "adequate power",
         power_verdict(101) == "adequate power",
         (Fraction(1 + 4, 101) < Fraction(1, 20)) is True, (Fraction(1 + 5, 101) < Fraction(1, 20)) is False,
         (5 * 2574 < 12870) is False, (5 * 2573 < 12870) is True]       # PBO < 0.20 at the boundary
    a2, b2 = auc_parts(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1.0]))
    t.append((a2, b2) == (6, 8))                                        # AUC 0.75 exactly
    ok = all(t)
    if not quiet:
        print("SELFTEST", "PASS" if ok else "FAIL", t)
    return ok


def summarize(out):
    rows = pd.read_csv(os.path.join(out, "rows.csv"))
    need(len(rows) == REPS * len(ARMS), f"rows {len(rows)}")
    dec = {"prereg_sha256": sha256(PREREG), "run_py_sha256": sha256(os.path.abspath(__file__)),
           "rows_sha256": sha256(os.path.join(out, "rows.csv")), "gates": {}, "verdicts": {},
           "descriptive": {}}
    for g in GATES + ("G2c",):
        dec["gates"][g] = {}
        for arm in ARMS:
            k = int(rows.loc[rows.arm == arm, g].sum())
            lo, hi = wilson(k, REPS)
            dec["gates"][g][arm] = {"k": k, "n": REPS, "rate": k / REPS, "wilson95": [lo, hi]}
    for g in GATES:
        dec["verdicts"][g] = {
            "size_N1": size_verdict(dec["gates"][g]["N1"]["k"]),
            "size_N0_drift_kept": size_verdict(dec["gates"][g]["N0"]["k"]),
            "power_P2": power_verdict(dec["gates"][g]["P2"]["k"]),
        }
    cost_cols = [c for c in rows.columns if c.startswith("G4_cost_")]
    dec["descriptive"]["G4_by_cost"] = {c: {arm: int(rows.loc[rows.arm == arm, c].sum()) for arm in ARMS}
                                        for c in cost_cols}
    dec["descriptive"]["mean_PBO"] = {arm: float((rows.pbo_k / rows.pbo_K)[rows.arm == arm].mean()) for arm in ARMS}
    dec["descriptive"]["mean_AUC_selected"] = {arm: float(rows.auc_sel[rows.arm == arm].mean()) for arm in ARMS}
    dec["descriptive"]["selected_config_counts"] = {
        arm: rows.loc[rows.arm == arm, "sel_config"].value_counts().sort_index().to_dict() for arm in ARMS}
    write_json(os.path.join(out, "decision.json"), dec)
    for g in GATES:
        print(g, {a: dec["gates"][g][a]["k"] for a in ARMS}, dec["verdicts"][g])
    return dec


# ---------------------------------------------------------------- commands
def git(*a):
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True)


def cmd_smoke(out):
    t0 = time.time()
    for arm in ARMS:
        for rep in range(5):
            evaluate((arm, 10_000 + rep))                # development replications, outside 0..199
    per = (time.time() - t0) / (5 * len(ARMS))
    res = {"seconds_per_series": round(per, 3), "projected_minutes_2_workers": round(per * REPS * len(ARMS) / 2 / 60, 1),
           "at": now(), "prereg_sha256": sha256(PREREG), "run_py_sha256": sha256(os.path.abspath(__file__))}
    write_json(os.path.join(out, "smoke.json"), res)
    print(json.dumps(res))


def cmd_run(out, workers):
    for f in ("PREREGISTRATION.md", "run.py"):
        need(git("ls-files", "--error-unmatch", f).returncode == 0, f"{f} is not committed")
    need(git("diff", "--quiet", "HEAD", "--", "PREREGISTRATION.md", "run.py").returncode == 0,
         "PREREGISTRATION.md or run.py differs from the committed version")
    commit = git("log", "-1", "--format=%H", "--", "PREREGISTRATION.md").stdout.strip()
    need(selftest(quiet=True), "selftest failed")
    lock = os.path.join(out, "lock.json")
    rec = {"prereg_commit": commit, "prereg_sha256": sha256(PREREG), "run_py_sha256": sha256(os.path.abspath(__file__)),
           "start": now(), "workers": workers}
    os.makedirs(out, exist_ok=True)
    need(not os.path.exists(lock), "lock.json exists: this output folder was already used")
    write_json(lock, rec)
    t0 = time.time()
    tasks = [(arm, rep) for arm in ARMS for rep in range(REPS)]
    with Pool(workers) as pool:
        rows = list(pool.imap(evaluate, tasks, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "rows.csv"), index=False)
    rec.update(end=now(), wall_s=round(time.time() - t0, 1), rows=len(df),
               rows_sha256=sha256(os.path.join(out, "rows.csv")))
    write_json(os.path.join(out, "run_log.json"), rec)
    summarize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "smoke", "run", "summarize"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    need(1 <= a.workers <= 2, "1 or 2 workers")
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    need(a.out, "--out is required")
    {"smoke": lambda: cmd_smoke(a.out), "run": lambda: cmd_run(a.out, a.workers),
     "summarize": lambda: summarize(a.out)}[a.cmd]()


if __name__ == "__main__":
    main()

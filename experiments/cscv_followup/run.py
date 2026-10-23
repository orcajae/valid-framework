"""E18 follow-up to E17: PBO with its companion statistic (probability of loss) and CSCV with an embargo.

Design: experiments/cscv_followup/PREREGISTRATION.md (committed to origin/main before any registered series is run).
Layer 1 needs numpy, scipy and pandas only. Layer 2 reads E17's private per-event matrices (stage3_R.npz).

Commands (run from the repository root with python3 -I -B; at most 2 workers):
  run.py unit      --out OUT [--e17 results/cscv_calibration]      unit tests  -> OUT/checks/unit.json
  run.py time      --out OUT                                        timing      -> OUT/checks/timing.json
  run.py selftest                                                   decision rules on fabricated counts
  run.py layer1    --out OUT                                        registered Layer 1 run (gated)
  run.py layer2    --out OUT --r PATH/stage3_R.npz [--e17 ...]      Layer 2 (gated; identity check first)
  run.py summarize --out OUT                                        rates and decisions F1-F5
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from itertools import combinations, product  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr, stdtrit  # noqa: E402
from scipy.stats import binom, rankdata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PREREG = os.path.join(HERE, "PREREGISTRATION.md")
PREREG_REL = "experiments/cscv_followup/PREREGISTRATION.md"
RUN_REL = "experiments/cscv_followup/run.py"
FROZEN = ("run_py_sha256", "prereg_body_sha256")            # compared by every gate; the Change log may grow
EPS = 1e-10
TS, MS, SS, RHOS, SERIALS = (250, 1000, 2500), (4, 8, 16, 32), (8, 16), (0.0, 0.5, 0.9), ("IID", "GARCH", "MA10")
CELLS = list(product(TS, MS, SS, RHOS, SERIALS))            # c = 0..215, last factor fastest (as E17)
REF = CELLS.index((1000, 8, 16, 0.5, "GARCH"))              # 103
ARM1 = {"N1": 1, "N2": 2, "N3": 3, "P1": 4, "P2": 5, "P3": 6}
REPS = {"N1": 1000, "N2": 1000, "N3": 500, "P1": 500, "P2": 500, "P3": 500}
EDGE = {"P1": 0.5, "P2": 1.0, "P3": 2.0}
SIGMA, ANN, BURN = 0.01, 252, 500
NS, DEV = 18, 99                                            # registered seed prefix; development prefix
E1, E2 = 9, 20                                              # embargo: Layer 1 (MA10 lags), Layer 2 (events)
COLS = {4: [0, 4, 8, 12], 8: [0, 2, 4, 6, 8, 10, 12, 14], 16: list(range(16)), 24: list(range(24))}
GRID = (0.05, 0.10, 0.20, 0.50)
TAU, PI = 0.20, 0.05                                        # primary screen C(0.20, 0.05)
LIMIT_H = 6.0


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


def body_sha(path):
    """sha256 of the pre-registration text before its Change log (changes logged after a launch are allowed)."""
    t = open(path, encoding="utf-8").read()
    i = t.find("\n## Change log")
    need(i >= 0, "the pre-registration has no Change log heading")
    return hashlib.sha256(t[:i].encode("utf-8")).hexdigest()


def shas():
    return {"run_py_sha256": sha256(os.path.abspath(__file__)), "prereg_body_sha256": body_sha(PREREG),
            "prereg_sha256": sha256(PREREG)}


def same(rec, cur):
    return all(rec.get(k) == cur[k] for k in FROZEN)


# ---------------------------------------------------------------- generator (E17 §1; identity tested against E17)
def equi(z, rho):
    return math.sqrt(rho) * z[:, :1] + math.sqrt(1 - rho) * z[:, 1:]


def shocks(rng, T, M, rho, serial):
    if serial == "IID":
        return equi(rng.standard_normal((T, M + 1)), rho)
    if serial == "MA10":
        xi = equi(rng.standard_normal((T + 9, M + 1)), rho)
        e = xi[9:].copy()
        for k in range(1, 10):
            e += xi[9 - k:9 - k + T]
        return e / math.sqrt(10)
    eta = rng.standard_t(5, size=T + BURN) / math.sqrt(5 / 3)
    h = [1.0] * (T + BURN)
    et = eta.tolist()
    for t in range(1, T + BURN):
        u = math.sqrt(h[t - 1]) * et[t - 1]
        h[t] = 0.02 + 0.08 * u * u + 0.90 * h[t - 1]
    g = equi(rng.standard_normal((T, M + 1)), rho)
    q = stdtrit(5, ndtr(-np.abs(g)))
    eps = np.where(g > 0, -q, q) / math.sqrt(5 / 3)
    return np.sqrt(np.array(h[BURN:]))[:, None] * eps


def means(arm, M):
    m = np.arange(M)
    if arm == "N1":
        sr = np.zeros(M)
    elif arm == "N2":
        sr = -1.0 * m / (M - 1)
    elif arm == "N3":
        mp = np.where(m == 0, EDGE["P2"], 0.0) / math.sqrt(ANN) * SIGMA
        return mp - mp[0]                                   # E17: the means of P2 lowered by the edge
    else:
        sr = np.where(m == 0, EDGE[arm], 0.0)
    return sr / math.sqrt(ANN) * SIGMA


def series(arm, c, j, prefix):
    T, M, S, rho, serial = CELLS[c]
    a = ARM1["P2" if arm == "N3" else arm]
    e = shocks(np.random.default_rng(list(prefix) + [a, c, j]), T, M, rho, serial)
    return means(arm, M) + SIGMA * e


def prefix_of(dev):
    return [DEV, NS] if dev else [NS]


# ---------------------------------------------------------------- CSCV, PoL, embargo (prereg §3)
_HALVES = {}


def halves(S):
    if S not in _HALVES:
        cs = list(combinations(range(S), S // 2))
        H = np.zeros((len(cs), S))
        for i, c in enumerate(cs):
            H[i, list(c)] = 1.0
        _HALVES[S] = H
    return _HALVES[S]


def block_stats(R, S, e=0):
    blocks = [b[e:] for b in np.array_split(np.arange(R.shape[0]), S)]
    n = np.array([len(b) for b in blocks], dtype=np.int64)
    need((n >= 2).all(), f"embargo leaves fewer than 2 rows in a block (T={R.shape[0]}, S={S}, e={e})")
    return n, np.stack([R[b].sum(0) for b in blocks]), np.stack([(R[b] ** 2).sum(0) for b in blocks])


def cscv_from_blocks(n, s1, s2, S, loss=True):
    """K, k (splits with logit(rank/(M+1)) <= 0, first-listed in-sample best) and, if loss, the number of splits
    whose selected configuration has an out-of-sample Sharpe ratio below 0."""
    n = n.astype(float)
    ind = halves(S)

    def sr(J):
        nn = (J @ n)[:, None]
        mu = (J @ s1) / nn
        return mu / (np.sqrt(np.maximum((J @ s2) / nn - mu * mu, 0.0)) + EPS)

    isr, oos = sr(ind), sr(1.0 - ind)                       # K x M
    K, M = isr.shape
    best = isr.argmax(1)
    ob = oos[np.arange(K), best]
    rank = (oos < ob[:, None]).sum(1) + ((oos == ob[:, None]).sum(1) + 1) / 2
    w = rank / (M + 1)
    k = int(np.sum(np.log(w / (1 - w)) <= 0))
    return (K, k, int(np.sum(ob < 0))) if loss else (K, k)


def cscv(R, S, e=0, loss=True):
    return cscv_from_blocks(*block_stats(R, S, e), S, loss=loss)


# ---------------------------------------------------------------- gates (prereg §5)
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def gate(out, stage):
    """Prereg §5: checks passed with these files; run.py and the pre-registration committed and identical to
    origin/main; the lock freezes run.py and the pre-registration body at the first launch."""
    cur = shas()
    for name in ("unit", "timing"):
        p = os.path.join(out, "checks", f"{name}.json")
        need(os.path.exists(p), f"{name} check missing: run `run.py {'unit' if name == 'unit' else 'time'} --out OUT`")
        c = json.load(open(p))
        need(same(c, cur), f"{name} check was made with other files")
        need(c.get("ok") is True, f"{name} check did not pass")
    for rel in (PREREG_REL, RUN_REL):
        need(git("ls-files", "--error-unmatch", rel).returncode == 0, f"{rel} is not committed")
    need(git("diff", "--quiet", "origin/main", "--", PREREG_REL, RUN_REL).returncode == 0,
         "run.py or the pre-registration differs from origin/main (commit, push, git fetch)")
    last = git("log", "-1", "--format=%H", "origin/main", "--", PREREG_REL).stdout.strip()
    need(bool(last), "no commit of the pre-registration on origin/main")
    lock = os.path.join(out, "checks", "lock.json")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    rec = {**cur, "prereg_commit": last, "first_launch": now(), "stage": stage}
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=1)
    except FileExistsError:
        need(same(json.load(open(lock)), cur),
             "lock.json holds other hashes: run.py or the pre-registration body changed after the first launch")
    return {**cur, "prereg_commit": last}


def acquire(out):
    """One run at a time in OUT (Layer 1 and Layer 2 together stay within two processes). A lock left by a dead
    process on this host is cleared."""
    p = os.path.join(out, "checks", "run.running")
    me = {"pid": os.getpid(), "host": socket.gethostname(), "start": now()}
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                json.dump(me, f)
            return p
        except FileExistsError:
            try:
                old = json.load(open(p))
            except (OSError, ValueError):
                old = {}
            alive = True
            if old.get("host") == me["host"]:
                try:
                    os.kill(int(old["pid"]), 0)
                except ProcessLookupError:
                    alive = False
                except (PermissionError, KeyError, ValueError, TypeError):
                    pass
            need(not alive, f"another run may be active in OUT ({p}: {old}). If no run.py process is running, "
                            "the lock is stale (an empty file, or a host name that changed): remove it after checking")
            os.remove(p)
            print(f"cleared a stale run lock of pid {old.get('pid')}", flush=True)
    need(False, "could not take the run lock")


def release(p):
    try:
        if json.load(open(p)).get("pid") == os.getpid():
            os.remove(p)
    except (OSError, ValueError):
        pass


def launch_log(out, stage, extra):
    p = os.path.join(out, "logs", f"{stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json")
    rec = {"stage": stage, "start": now(), "pid": os.getpid(), "host": socket.gethostname(),
           "stop_conditions": ["a unit test fails", "the timing projection exceeds the limit",
                               "the Layer 2 identity fails", "a registered series fails to compute"],
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
           **extra}
    write_json(p, rec)
    return p, rec


# ---------------------------------------------------------------- Layer 1
COLS1 = ["cell", "T", "M", "S", "rho", "serial", "arm", "j", "K", "k", "loss", "kE", "lossE"]


def l1_cell(args):
    """Pool task. A STOP inside a worker (SystemExit) is turned into an error the parent sees, so the run stops."""
    try:
        return _l1_cell(*args)
    except SystemExit as e:
        raise RuntimeError(str(e)) from None


def _l1_cell(c, dev, nmax):
    T, M, S, rho, serial = CELLS[c]
    rows = []
    for arm, reps in REPS.items():
        for j in range(min(reps, nmax) if nmax else reps):
            R = series(arm, c, j, prefix_of(dev))
            K, k, loss = cscv(R, S, 0)
            KE, kE, lossE = cscv(R, S, E1)
            need(K == KE, "split counts differ")
            rows.append((c, T, M, S, rho, serial, arm, j, K, k, loss, kE, lossE))
    return rows


def cmd_layer1(out, workers):
    g = gate(out, "layer1")
    lk = acquire(out)
    try:
        _layer1(out, workers, g)
    finally:
        release(lk)


def _layer1(out, workers, g):
    tim = json.load(open(os.path.join(out, "checks", "timing.json")))
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    todo = [c for c in range(len(CELLS)) if not os.path.exists(os.path.join(parts, f"l1_c{c:03d}.csv"))]
    t0 = time.time()
    exp_end = datetime.fromtimestamp(t0 + tim["projected_h"] * 3600 * len(todo) / len(CELLS), timezone.utc).isoformat()
    logp, log = launch_log(out, "layer1", {**g, "cells_todo": len(todo), "workers": workers, "expected_end": exp_end})
    print(json.dumps({k: log[k] for k in ("start", "expected_end", "cells_todo")}), flush=True)
    with Pool(workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(l1_cell, [(c, False, None) for c in todo])):
            c = rows[0][0]
            tmp = os.path.join(parts, f"l1_c{c:03d}.csv.{os.getpid()}.tmp")
            pd.DataFrame(rows, columns=COLS1).to_csv(tmp, index=False)
            os.replace(tmp, os.path.join(parts, f"l1_c{c:03d}.csv"))
            el = time.time() - t0
            print(f"[layer1] {i + 1}/{len(todo)} cells · {el / 60:.1f} min · eta "
                  f"{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
    files = [os.path.join(parts, f"l1_c{c:03d}.csv") for c in range(len(CELLS))]
    need(all(os.path.exists(f) for f in files), "missing Layer 1 parts")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    need(len(df) == 216 * sum(REPS.values()), f"Layer 1 rows {len(df)}")
    dst = os.path.join(out, "layer1_rows.csv.gz")
    tmp = f"{dst}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    os.replace(tmp, dst)
    log.update(end=now(), wall_s=round(time.time() - t0, 1), rows=len(df), rows_sha256=sha256(dst))
    write_json(logp, log)
    write_json(os.path.join(out, "checks", "layer1_done.json"),
               {"rows": len(df), "rows_sha256": sha256(dst), "at": now(), **g})
    print("layer1 done", len(df), flush=True)


# ---------------------------------------------------------------- Layer 2 (secondary analysis of E17's matrices)
COLS2 = ["arm", "seed", "M", "S", "e", "K", "k", "loss"]


def cmd_layer2(rpath, e17, out):
    g = gate(out, "layer2")
    lk = acquire(out)
    try:
        _layer2(rpath, e17, out, g)
    finally:
        release(lk)


def _layer2(rpath, e17, out, g):
    logp, log = launch_log(out, "layer2", {**g, "stage3_R_sha256": sha256(rpath),
                                           "expected_end": "written after the first 20 series"})
    z = np.load(rpath)
    bs = np.load(os.path.join(e17, "stage3_blockstats.npz"))
    # identity with E17's published block statistics, before any new statistic
    n_ok = 0
    for key in sorted(z.files):
        arm, seed, tag = key.rsplit("_", 2)
        need(tag == "R", f"unexpected key {key}")
        if arm == "R":
            continue
        R = z[key]
        for S in (8, 16):
            n, s1, s2 = block_stats(R, S, 0)
            p = f"{arm}_{seed}_S{S}"
            need(np.array_equal(n, bs[p + "_n"]) and np.array_equal(s1, bs[p + "_sum"])
                 and np.array_equal(s2, bs[p + "_sumsq"]), f"Layer 2 identity fails for {p}")
        n_ok += 1
    need(n_ok == 1500, f"Layer 2 identity covered {n_ok} series")
    # arm R: its eight canonical PBO counts published in E17's stage3_rows.csv (PBO count only)
    s3 = pd.read_csv(os.path.join(e17, "stage3_rows.csv"))
    r_keys = [k for k in z.files if k.rsplit("_", 2)[0] == "R"]
    need(len(r_keys) == 1, f"arm R keys: {r_keys}")
    seed_r = r_keys[0].rsplit("_", 2)[1]
    row = s3[(s3.arm == "R") & (s3.seed.astype(str) == seed_r)]
    need(len(row) == 1, "arm R row in stage3_rows.csv")
    row = row.iloc[0]
    for M in COLS:
        for S in (8, 16):
            K, k = cscv(z[r_keys[0]][:, COLS[M]], S, 0, loss=False)
            need(K == int(row[f"M{M}_S{S}_K"]) and k == int(row[f"M{M}_S{S}_k_first"]),
                 f"Layer 2 identity fails for arm R (M {M}, S {S})")
    write_json(os.path.join(out, "checks", "layer2_identity.json"),
               {"series": n_ok, "mismatch": 0, "arm_R_pbo_counts": 8, "arm_R_mismatch": 0,
                "stage3_R_sha256": log["stage3_R_sha256"], "at": now(), **g})
    rows, trim = [], {}
    t0 = time.time()
    for i, key in enumerate(sorted(z.files)):
        if i == 20:
            log["expected_end"] = datetime.fromtimestamp(
                t0 + (time.time() - t0) / 20 * len(z.files), timezone.utc).isoformat()
            write_json(logp, log)
            print(f"[layer2] expected end {log['expected_end']}", flush=True)
        if i % 250 == 0:
            print(f"[layer2] {i}/{len(z.files)} series · {(time.time() - t0) / 60:.1f} min", flush=True)
        arm, seed, _ = key.rsplit("_", 2)
        R = z[key]
        need(np.isfinite(R).all(), f"non-finite values in {key}")
        for S in (8, 16):
            nE, s1E, s2E = block_stats(R, S, E2)
            if arm != "R":
                trim[f"{arm}_{seed}_S{S}_e{E2}_n"] = nE
                trim[f"{arm}_{seed}_S{S}_e{E2}_sum"] = s1E
                trim[f"{arm}_{seed}_S{S}_e{E2}_sumsq"] = s2E
            for M in COLS:
                for e in (0, E2):
                    K, k, loss = cscv(R[:, COLS[M]], S, e)
                    rows.append((arm, seed, M, S, e, K, k, loss))
    df = pd.DataFrame(rows, columns=COLS2)
    need(len(df) == len(z.files) * 2 * len(COLS) * 2, f"Layer 2 rows {len(df)}")
    dst = os.path.join(out, "layer2_rows.csv")
    tmp = f"{dst}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, dst)
    npz = os.path.join(out, "layer2_blockstats_embargo.npz")
    tmp = f"{npz[:-4]}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **trim)
    os.replace(tmp, npz)
    log.update(end=now(), wall_s=round(time.time() - t0, 1), rows=len(df), rows_sha256=sha256(dst),
               blockstats_sha256=sha256(npz))
    write_json(logp, log)
    write_json(os.path.join(out, "checks", "layer2_done.json"),
               {"rows": len(df), "rows_sha256": sha256(dst), "blockstats_sha256": sha256(npz), "at": now(), **g})
    print("layer2 done", len(df), flush=True)


# ---------------------------------------------------------------- unit tests (prereg §5)
def e15_cscv_matrices(R, S):
    """E15 experiments/inbox7/run.py cscv_matrices, copied verbatim (EPS = 1e-10)."""
    blocks = np.array_split(np.arange(R.shape[0]), S)
    nb = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([R[b].sum(0) for b in blocks])
    s2 = np.stack([(R[b] ** 2).sum(0) for b in blocks])
    ind = halves(S)

    def sr(I):
        n = (I @ nb)[:, None]
        mu = (I @ s1) / n
        var = np.maximum((I @ s2) / n - mu ** 2, 0.0)
        return mu / (np.sqrt(var) + EPS)

    return sr(ind).T, sr(1.0 - ind).T


def e15_pbo_fast(is_m, oos_m):
    """E15 pbo_fast, copied verbatim."""
    M = is_m.shape[0]
    best = is_m.argmax(0)
    ranks = rankdata(oos_m, axis=0)
    w = ranks[best, np.arange(is_m.shape[1])] / (M + 1)
    return float(np.mean(np.log(w / (1 - w)) <= 0))


def direct_loop(R, S, e):
    """PBO and loss counts by a direct loop over splits, from the rows themselves."""
    blocks = [b[e:] for b in np.array_split(np.arange(R.shape[0]), S)]
    M = R.shape[1]
    k = loss = 0
    cs = list(combinations(range(S), S // 2))
    for c in cs:
        i_rows = np.concatenate([blocks[b] for b in c])
        o_rows = np.concatenate([blocks[b] for b in range(S) if b not in c])
        isr = R[i_rows].mean(0) / (R[i_rows].std(0) + EPS)
        osr = R[o_rows].mean(0) / (R[o_rows].std(0) + EPS)
        b = int(np.argmax(isr))
        w = rankdata(osr)[b] / (M + 1)
        k += int(math.log(w / (1 - w)) <= 0)
        loss += int(osr[b] < 0)
    return len(cs), k, loss


def cmd_unit(e17, out):
    res = {}
    r1 = pd.read_csv(os.path.join(e17, "stage1_rows.csv.gz"), usecols=["cell", "arm", "j", "K", "k_first"])
    samp = r1[r1.cell < 216].sample(300, random_state=20261024)
    bad = 0
    for _, row in samp.iterrows():
        R = series(row.arm, int(row.cell), int(row.j), [])          # E17's seeds [a, c, j]
        K, k = cscv(R, CELLS[int(row.cell)][2], 0, loss=False)       # PBO count only (prereg §5)
        bad += (K != row.K) or (k != row.k_first)
    res["generator_identity_E17"] = {"n": 300, "mismatch": int(bad)}
    bad = n = 0
    for serial, S, M, T in product(SERIALS, (8, 16), (4, 8), (250, 1000)):
        c = CELLS.index((T, M, S, 0.5, serial))
        for j in range(5):
            R = series("N1", c, 10_000 + j, prefix_of(True))
            K, k = cscv(R, S, 0, loss=False)
            im, om = e15_cscv_matrices(R, S)
            bad += Fraction(k, K) != Fraction(e15_pbo_fast(im, om)).limit_denominator(K)
            n += 1
    res["cscv_identity_E15"] = {"n": n, "mismatch": int(bad)}
    bad = n = 0
    for serial, S in product(SERIALS, (8, 16)):
        c = CELLS.index((250, 8, S, 0.5, serial))
        for j in range(3):
            for arm in ("N1", "N2", "P2"):
                R = series(arm, c, 20_000 + j, prefix_of(True))
                for e in (0, E1):
                    bad += cscv(R, S, e) != direct_loop(R, S, e)
                    n += 1
    res["direct_loop"] = {"n": n, "mismatch": int(bad)}
    gap_ok, left = True, []
    for T, S in product(TS, SS):
        bl = [b[E1:] for b in np.array_split(np.arange(T), S)]
        left += [len(b) for b in bl]
        for a, b in zip(bl[:-1], bl[1:]):
            gap_ok &= int(b.min() - a.max()) >= E1 + 1
    res["embargo_spacing"] = {"ok": bool(gap_ok), "min_rows_left": int(min(left))}
    res["decision_selftest"] = {"ok": bool(selftest(quiet=True))}
    ok = (all(v.get("mismatch", 0) == 0 for v in res.values() if "mismatch" in v)
          and res["embargo_spacing"]["ok"] and res["decision_selftest"]["ok"])
    print(json.dumps(res, indent=1))
    print("UNIT PASS" if ok else "UNIT FAIL")
    if out:
        write_json(os.path.join(out, "checks", "unit.json"), {"ok": bool(ok), "at": now(), **res, **shas()})
    return ok


# ---------------------------------------------------------------- timing (prereg §5)
def cmd_time(out, workers):
    per = {}
    for c in range(len(CELLS)):
        T, M, S, rho, serial = CELLS[c]
        if (T, M, S, serial) in per:
            continue
        t0 = time.time()
        l1_cell((c, True, 3))                                   # 3 series of each arm, development seeds
        per[(T, M, S, serial)] = (time.time() - t0) / (3 * len(REPS))
    tot = sum(per[(T, M, S, serial)] * sum(REPS.values()) for T, M, S, rho, serial in CELLS)
    proj = tot / workers / 3600
    res = {"projected_h": round(proj, 2), "workers": workers, "limit_h": LIMIT_H, "ok": bool(proj <= LIMIT_H),
           "at": now(), **shas()}
    print(json.dumps(res))
    if out:
        write_json(os.path.join(out, "checks", "timing.json"), res)
    return res


# ---------------------------------------------------------------- decisions (prereg §4; exact fractions)
def pp(k, n):
    return Fraction(100 * int(k), int(n))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(b, c), n, 0.5)))


def rule_f1(k_p2, k_n3, b, c, n=500):
    need(b - c == k_p2 - k_n3 and 0 <= b <= k_p2 and 0 <= c <= k_n3 and b + c <= n, "F1 counts are inconsistent")
    diff = pp(k_p2, n) - pp(k_n3, n)
    p_m = mcnemar_exact(b, c)
    if diff >= 10 and p_m < 0.05:
        return "separates", diff, p_m
    if abs(diff) < 5:
        return "does not separate", diff, p_m
    return "unclear", diff, p_m


def rule_f3(cnt):
    """cnt: {arm: (k, n)} of C(0.20, 0.05) in the reference cell."""
    ok = all(pp(*cnt[a]) <= 10 for a in ("N1", "N2", "N3"))
    if not ok:
        return "not useful"
    return "useful" if pp(*cnt["P2"]) >= 50 else "safe but weak"


def rule_f4(ex0, exE):
    if ex0 < 3:
        return "premise not reproduced"
    if abs(exE) <= 1:
        return "restores"
    if exE < -1:
        return "overshoots"
    if exE >= 3:
        return "does not restore"
    return "partial"


def r95_counts(n, pbar, cells, draws=10_000):
    """R95 in counts, exact: numpy's 'linear' 95th percentile of the integer spans, interpolated as a Fraction."""
    s = np.random.default_rng(0).binomial(n, pbar, size=(draws, cells))
    sp = np.sort(s.max(1) - s.min(1))
    pos = Fraction(95 * (draws - 1), 100)
    i = int(pos)
    return int(sp[i]) + (pos - i) * int(sp[min(i + 1, draws - 1)] - sp[i])


def rule_f2(per_arm, verdict=True):
    """per_arm: {arm: {cell: (k, n)}} of L(0.05). Level holds if, for each arm, the pooled Wilson upper bound
    <= 6.5% (E17 D3's bound) and span - R95 <= 2 percentage points (E17 D1's bound), R95 over as many cells.
    span and R95 are compared exactly, in counts. verdict=False reports the same quantities without a verdict."""
    out, ok = {}, True
    for arm, cells in per_arm.items():
        n1 = REPS[arm]
        need(all(v[1] == n1 for v in cells.values()), f"F2 cell sizes {arm}")
        ks = [v[0] for v in cells.values()]
        k, n = sum(ks), n1 * len(ks)
        span_c = max(ks) - min(ks)
        R = r95_counts(n1, k / n, len(ks))
        excess = Fraction(100, n1) * (span_c - R)
        hi = wilson(k, n)[1]
        rec = {"cells": len(ks), "k": k, "n": n, "pooled": k / n, "wilson_hi": hi,
               "span_pp": float(Fraction(100 * span_c, n1)), "R95_pp": float(R * Fraction(100, n1)),
               "span_minus_R95_pp": float(excess)}
        if verdict:
            a_ok = (hi <= 0.065) and (excess <= 2)
            ok &= a_ok
            rec["ok"] = bool(a_ok)
        out[arm] = rec
    return (("level holds" if ok else "level fails") if verdict else None), out


def selftest(quiet=False):
    """Decision rules on fabricated counts (no generator): exact boundaries of F1, F2, F3, F4."""
    t = []
    t.append(rule_f1(300, 250, 250, 200)[0] == "separates")            # exactly 10 points, p_M = 0.0208
    t.append(rule_f1(275, 250, 50, 25)[0] == "unclear")                 # exactly 5 points
    t.append(rule_f1(270, 250, 40, 20)[0] == "does not separate")       # 4 points
    t.append(rule_f1(250, 300, 200, 250)[0] == "unclear")               # -10 points
    t.append(rule_f3({"N1": (50, 500), "N2": (100, 1000), "N3": (50, 500), "P2": (250, 500)}) == "useful")
    t.append(rule_f3({"N1": (50, 500), "N2": (101, 1000), "N3": (50, 500), "P2": (400, 500)}) == "not useful")
    t.append(rule_f3({"N1": (0, 500), "N2": (0, 1000), "N3": (0, 500), "P2": (249, 500)}) == "safe but weak")
    t.append(rule_f4(pp(720, 24000), pp(240, 24000)) == "restores")    # exactly 3.0, exactly 1.0
    t.append(rule_f4(pp(720, 24000), pp(720, 24000)) == "does not restore")
    t.append(rule_f4(pp(719, 24000), pp(0, 24000)) == "premise not reproduced")
    t.append(rule_f4(pp(1000, 24000), pp(-241, 24000)) == "overshoots")
    t.append(rule_f4(pp(1000, 24000), pp(241, 24000)) == "partial")
    t.append(rule_f4(pp(1000, 24000), pp(-240, 24000)) == "restores")  # exactly -1.0
    t.append(rule_f4(pp(1000, 24000), pp(719, 24000)) == "partial")
    hold = {a: {c: (50 if a != "N3" else 25, REPS[a]) for c in range(144)} for a in ("N1", "N2", "N3")}
    t.append(rule_f2(hold)[0] == "level holds")                        # 5.0% everywhere, span 0
    fail = {a: {c: (70 if a != "N3" else 35, REPS[a]) for c in range(144)} for a in ("N1", "N2", "N3")}
    t.append(rule_f2(fail)[0] == "level fails")                        # 7.0% pooled
    wide = {a: {c: ((10 if c % 2 else 90) if a != "N3" else (5 if c % 2 else 45), REPS[a]) for c in range(144)}
            for a in ("N1", "N2", "N3")}
    t.append(rule_f2(wide)[0] == "level fails")                        # 5.0% pooled, span 8 points
    t.append(r95_counts(500, 0.05, 144) == 31)
    tie = {c: [25, 500] for c in range(144)}
    tie[0][0] += 21
    tie[1][0] -= 20
    tie[2][0] -= 1
    t.append(rule_f2({"N3": {c: tuple(v) for c, v in tie.items()}})[0] == "level holds")  # span - R95 = 2.0
    tie[0][0] += 1
    tie[2][0] -= 1
    t.append(rule_f2({"N3": {c: tuple(v) for c, v in tie.items()}})[0] == "level fails")  # 2.2 points
    t.append(abs(mcnemar_exact(250, 200) - 0.020792) < 1e-5)
    ok = all(t)
    if not quiet:
        print("SELFTEST", "PASS" if ok else "FAIL", t)
    return ok


def screens(df, kcol, lcol):
    """P(t), L(t) and C(t, u) as exact integer comparisons: k/K < t  <=>  100 k < (100 t) K."""
    out = {}
    for t in GRID:
        out[f"P{t:.2f}"] = df[kcol] * 100 < round(t * 100) * df.K
        out[f"L{t:.2f}"] = df[lcol] * 100 < round(t * 100) * df.K
    for t in GRID:
        for u in GRID:
            out[f"C{t:.2f}_{u:.2f}"] = out[f"P{t:.2f}"] & out[f"L{u:.2f}"]
    return out


def cmd_summarize(out):
    cur = shas()
    done = json.load(open(os.path.join(out, "checks", "layer1_done.json")))
    need(same(done, cur), "run.py or the pre-registration body changed after Layer 1 was run")
    rows_path = os.path.join(out, "layer1_rows.csv.gz")
    need(sha256(rows_path) == done["rows_sha256"], "layer1_rows.csv.gz differs from the run's record")
    df = pd.read_csv(rows_path)
    rates = []
    for variant, kc, lc in (("canonical", "k", "loss"), ("embargoed", "kE", "lossE")):
        sc = screens(df, kc, lc)
        for name, m in sc.items():
            g = m.groupby([df.cell, df.arm]).agg(["sum", "count"])
            for (c, a), (k, n) in g.iterrows():
                lo, hi = wilson(int(k), int(n))
                rates.append((c, *CELLS[c], a, variant, name, int(k), int(n), k / n, lo, hi))
        mp = (df[kc] / df.K).groupby([df.cell, df.arm]).mean()
        ml = (df[lc] / df.K).groupby([df.cell, df.arm]).mean()
        for (c, a), v in mp.items():
            rates.append((c, *CELLS[c], a, variant, "mean_PBO", None, None, v, None, None))
            rates.append((c, *CELLS[c], a, variant, "mean_PoL", None, None, ml[(c, a)], None, None))
    rates = pd.DataFrame(rates, columns=["cell", "T", "M", "S", "rho", "serial", "arm", "variant", "statistic",
                                         "k", "n", "rate", "wilson_lo", "wilson_hi"])
    rates = rates.astype({"k": "Int64", "n": "Int64"})
    rates.to_csv(os.path.join(out, "rates_layer1.csv"), index=False)

    sc0 = screens(df, "k", "loss")
    prim = sc0[f"C{TAU:.2f}_{PI:.2f}"]
    ref = df.cell == REF
    p2 = df[ref & (df.arm == "P2")].sort_values("j")
    n3 = df[ref & (df.arm == "N3")].sort_values("j")
    need(p2.j.tolist() == n3.j.tolist() == list(range(500)), "reference pairs")
    a, b = prim[p2.index].values, prim[n3.index].values
    f1, diff, p_m = rule_f1(int(a.sum()), int(b.sum()), int((a & ~b).sum()), int((~a & b).sum()))
    def level_cells(l5, serials, ncell):
        per = {}
        for arm in ("N1", "N2", "N3"):
            sel = (df.arm == arm) & df.serial.isin(serials)
            x = l5[sel].groupby(df.cell[sel]).agg(["sum", "count"])
            per[arm] = {int(c): (int(k), int(n)) for c, (k, n) in x.iterrows()}
            need(len(per[arm]) == ncell and all(v[1] == REPS[arm] for v in per[arm].values()), f"F2 cells {arm}")
        return per

    f2, f2d = rule_f2(level_cells(sc0[f"L{PI:.2f}"], ("IID", "GARCH"), 144))     # F2: IID and GARCH cells
    f2_ma10 = {"canonical": rule_f2(level_cells(sc0[f"L{PI:.2f}"], ("MA10",), 72), verdict=False)[1],
               "embargoed": rule_f2(level_cells(screens(df, "kE", "lossE")[f"L{PI:.2f}"], ("MA10",), 72),
                                    verdict=False)[1]}
    cnt = {arm: (int(prim[ref & (df.arm == arm)].sum()), int((ref & (df.arm == arm)).sum())) for arm in REPS}
    f3 = rule_f3(cnt)

    def k_pool(serial, T, kcol):
        x = df[(df.arm == "N1") & (df.serial == serial) & (df["T"] == T)]
        need(len(x) == 24000, f"pool size {serial} {T}: {len(x)}")
        return int((5 * x[kcol] < x.K).sum())                 # PBO < 0.20 exactly

    k = {f"{s}_{v}": k_pool(s, 250, kc) for s in ("MA10", "IID", "GARCH") for v, kc in
         (("canonical", "k"), ("embargoed", "kE"))}
    ex0 = pp(k["MA10_canonical"] - k["IID_canonical"], 24000)
    exE = pp(k["MA10_embargoed"] - k["IID_embargoed"], 24000)
    f4 = rule_f4(ex0, exE)
    f5 = {}
    for serial, T in product(SERIALS, TS):
        x = (df.arm == "P2") & (df.serial == serial) & (df["T"] == T)
        f5[f"{serial}_T{T}"] = {}
        for v, kc, lc in (("canonical", "k", "loss"), ("embargoed", "kE", "lossE")):
            s_ = screens(df[x], kc, lc)
            f5[f"{serial}_T{T}"][f"P0.20_{v}"] = [int(s_["P0.20"].sum()), int(x.sum())]
            f5[f"{serial}_T{T}"][f"C_{v}"] = [int(s_[f"C{TAU:.2f}_{PI:.2f}"].sum()), int(x.sum())]
    x = ref & (df.arm == "P2")
    f5["reference_cell"] = {}
    for v, kc, lc in (("canonical", "k", "loss"), ("embargoed", "kE", "lossE")):
        s_ = screens(df[x], kc, lc)
        f5["reference_cell"][f"P0.20_{v}"] = [int(s_["P0.20"].sum()), int(x.sum())]
        f5["reference_cell"][f"C_{v}"] = [int(s_[f"C{TAU:.2f}_{PI:.2f}"].sum()), int(x.sum())]
    dec = {
        **cur, "layer1_rows_sha256": done["rows_sha256"],
        "F1": {"P2_pass": int(a.sum()), "N3_pass": int(b.sum()), "P2_only": int((a & ~b).sum()),
               "N3_only": int((~a & b).sum()), "P2_minus_N3_pp": float(diff), "p_M": p_m, "verdict": f1},
        "F2": {"cells": "IID and GARCH (144)", "arms": f2d, "verdict": f2,
               "MA10_beside_no_verdict": f2_ma10},
        "F3": {"counts": cnt, "verdict": f3},
        "F4": {"N1_PBO020_counts_T250_of_24000": k, "excess0_pp": float(ex0), "excessE_pp": float(exE),
               "verdict": f4},
        "F5": f5,
        "use_of_outcomes": "PREREGISTRATION.md section 4; the paper type assigned by E17 does not change",
    }
    l2 = os.path.join(out, "layer2_rows.csv")
    if os.path.exists(l2):
        d2p = os.path.join(out, "checks", "layer2_done.json")
        need(os.path.exists(d2p), "layer2_rows.csv without checks/layer2_done.json")
        d2d = json.load(open(d2p))
        need(same(d2d, cur) and sha256(l2) == d2d["rows_sha256"], "layer2_rows.csv differs from the run's record")
        dec["layer2_rows_sha256"] = d2d["rows_sha256"]
        d2 = pd.read_csv(l2, dtype={"seed": str})
        r2 = []
        for (arm, M, S, e), g in d2.groupby(["arm", "M", "S", "e"]):
            s_ = screens(g, "k", "loss")
            for name, m in s_.items():
                kk, nn = int(m.sum()), int(len(m))
                lo, hi = wilson(kk, nn)
                r2.append((arm, M, S, e, name, kk, nn, kk / nn, lo, hi))
            r2.append((arm, M, S, e, "mean_PBO", None, None, float((g.k / g.K).mean()), None, None))
            r2.append((arm, M, S, e, "mean_PoL", None, None, float((g.loss / g.K).mean()), None, None))
        pd.DataFrame(r2, columns=["arm", "M", "S", "e", "statistic", "k", "n", "rate", "wilson_lo", "wilson_hi"]
                     ).astype({"k": "Int64", "n": "Int64"}).to_csv(os.path.join(out, "rates_layer2.csv"), index=False)
        dec["layer2"] = "rates_layer2.csv (descriptive, no verdict)"
    write_json(os.path.join(out, "decision.json"), dec)
    print(json.dumps({kk: dec[kk]["verdict"] for kk in ("F1", "F2", "F3", "F4")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unit", "time", "selftest", "layer1", "layer2", "summarize"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--e17", default=os.path.join(ROOT, "results", "cscv_calibration"))
    ap.add_argument("--r", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    need(1 <= a.workers <= 2, "1 or 2 workers")
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "unit":
        sys.exit(0 if cmd_unit(a.e17, a.out) else 1)
    if a.cmd == "time":
        sys.exit(0 if cmd_time(a.out, a.workers)["ok"] else 1)
    need(a.out, "--out is required")
    if a.cmd == "layer1":
        cmd_layer1(a.out, a.workers)
    elif a.cmd == "layer2":
        need(a.r, "--r PATH/stage3_R.npz is required")
        cmd_layer2(a.r, a.e17, a.out)
    else:
        cmd_summarize(a.out)


if __name__ == "__main__":
    main()

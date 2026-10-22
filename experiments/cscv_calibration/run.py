"""Size and power of PBO thresholds, and a null-calibrated test of rank persistence.

Design: valid-framework experiments/cscv_calibration/PREREGISTRATION.md (public, 2ead385, file sha256
cac0d83448d3794d5570af4e70fb85f27641762c9a9420f8067f4e9f354ec19c). The text before "## Change log"
is checked against its frozen digest at every start. Nothing here may change the design; every
implementation detail the pre-registration leaves open is listed in IMPLEMENTATION_NOTES below.

E15 (experiments/inbox7/run.py) is imported unchanged: cscv_matrices, pbo_fast, build, cv6_returns,
gate, demean, make_sb_series. It imports E14 (experiments/true_pbo/run.py: make_series, prepare,
asset_returns, make_p) and the valid-framework worktree at 018f4ea (VF env var, default /tmp/vf-pbo).

Layer 1 (prereg section 1): T x M synthetic return matrices; 216 cells (itertools.product of T, M, S,
rho, serial, last factor fastest); arms N1, N2, N3, P1, P2, P3; cell 216 = the unequal-volatility
variant of the reference cell. Layer 2 (section 2): the E15 S3 pipeline with a 24-configuration
CatBoost grid; arms A-BTC, A-ETH, A-SOL, D, C, B0, B1, B2, F1, F2, R.
Statistics (section 3): PBO (three tie rules), calibrated p-value, Reality Check, L sensitivity.

usage (python3 -I, from the repository root):
  run.py unit                                     # section 6 unit tests + implementation tests
  run.py repl                                     # Layer 2 replication on E15 A-BTC seeds 0-4 (real data)
  run.py repl --dev --out DIR                     # proxy on synthetic data: 24-grid M = 8 columns == E15 S3 R
  run.py time   --stage 1|2|3                     # times and projections only, never a statistic
  run.py calib                                    # beta for F1 / F2 on seeds 2000-2019 -> beta_F.json
  run.py betalog [--append]                       # prints the Change-log beta lines; --append writes them
  run.py stage1 [--shard k/n] [--cells a-b] [--arms N1,N2,...]
  run.py stage2 [--shard k/n] [--cells a-b] [--arms N1,N2,N3,P2]
  run.py stage3 [--shard k/n] [--arms A-BTC,D,...] [--seeds a-b]   # --seeds: whole 10-seed chunks
options:
  --out DIR   output directory (default results/cscv_calibration); parts, checks, logs go under it
  --dev       development namespace: Layer 1 seeds prefixed with 99, Layer 2 RandomState seeds
              99,000,000 + s and streams prefixed with 99, synthetic data in Layer 2. --out must be
              outside results/. No registered series is run.
  --n N       (dev only) series per unit, for smoke tests
  --B B       (dev only) bootstrap resamples
  --append    (betalog) append the lines under "## Change log" (dev: to OUT/PREREGISTRATION.dev.md)
Two shards run in parallel at most (prereg: 2 workers): run.py stage1 --shard 0/2 & run.py stage1 --shard 1/2.
"""
import argparse
import contextlib
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_spec = importlib.util.spec_from_file_location("e15", os.path.join(HERE, "..", "inbox7", "run.py"))
e15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e15)  # imports E14 and the VF worktree at 018f4ea
e14 = e15.e14
np, pd = e15.np, e15.pd
from experiments import data_sources  # noqa: E402  (VF worktree, put on sys.path by E14)
from scipy import integrate  # noqa: E402
from scipy.special import ndtr, stdtrit  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402  (installed with scikit-learn)

PREREG = os.path.join(HERE, "PREREGISTRATION.md")
PREREG_COMMIT = "2ead3855749b3a2a314f647ad0f6fca2fd0429fe"
PREREG_BODY_SHA = "cf81e2099a6cdd08c831df0df0e0274838c6c0e0af3fd599db626bf92d7dd73a"
OUT_DEFAULT = os.path.join(ROOT, "results", "cscv_calibration")
EPS = e15.EPS

# ---------------------------------------------------------------- Layer 1 design (prereg section 1)
TS, MS, SS, RHOS, SERIALS = (250, 1000, 2500), (4, 8, 16, 32), (8, 16), (0.0, 0.5, 0.9), ("IID", "GARCH", "MA10")
CELLS = list(product(TS, MS, SS, RHOS, SERIALS))          # c = 0..215, last factor fastest
REF = CELLS.index((1000, 8, 16, 0.5, "GARCH"))            # reference cell (= 103)
UNEQ = 216                                                 # unequal-volatility variant (seed [a, 216, j])
ARM1 = {"N1": 1, "N2": 2, "N3": 3, "P1": 4, "P2": 5, "P3": 6}
REPS1 = {"N1": 1000, "N2": 1000, "N3": 500, "P1": 500, "P2": 500, "P3": 500}
UNEQ_REPS = {"N1": 1000, "N2": 1000, "P2": 500}
EDGE = {"P1": 0.5, "P2": 1.0, "P3": 2.0}                   # annual Sharpe ratio of configuration 1
SIGMA, ANN, BURN = 0.01, 252, 500
T5_SD = math.sqrt(5 / 3)                                   # sd of Student-t(5)
RHO_P_STATED = {0.0: 0.0, 0.5: 0.488, 0.9: 0.894}          # prereg text, 3 decimals
S2_CELLS = [c for c, (T, M, S, rho, _) in enumerate(CELLS)
            if T in (250, 1000) and M in (4, 8, 16) and S == 8 and rho == 0.5] + [REF]
S2_ARMS, S2_REPS = ("N1", "N2", "N3", "P2"), 500
B_REG, B_FALLBACK, L0, L_SENS = 199, 99, 20, (5, 50)
EST_H = {1: (3.0, 5.0), 2: (3.0, 5.0), 3: (4.0, 6.0)}      # prereg section 6, 2 workers; limit = 2 x upper end

# ---------------------------------------------------------------- Layer 2 design (prereg section 2)
GRID24 = [(d, lr, cw, l2) for d in (4, 6, 8) for lr in (0.03, 0.1) for cw in (None, "Balanced") for l2 in (3, 10)]
SUB = {4: [i for i, g in enumerate(GRID24) if g[0] in (4, 6) and g[2] is None and g[3] == 3],
       8: [i for i, g in enumerate(GRID24) if g[0] in (4, 6) and g[3] == 3],
       16: [i for i, g in enumerate(GRID24) if g[0] in (4, 6)],
       24: list(range(24))}
CALIB_MS = ((8, 16), (16, 8))                              # (M, S) of the calibrated p and the Reality Check
ARM2 = {"A-BTC": 1, "A-ETH": 2, "A-SOL": 3, "D": 4, "C": 5, "B0": 6, "B1": 7, "B2": 8, "F1": 9, "F2": 10, "R": 11}
SEEDS2 = {**{a: range(0, 200) for a in ("A-BTC", "A-ETH", "A-SOL", "D", "C")},
          **{a: range(1000, 1100) for a in ("B0", "B1", "B2", "F1", "F2")}, "R": range(0, 1)}
E15_ARM = {"A-BTC": ("A", "BTC"), "A-ETH": ("A", "ETH"), "A-SOL": ("A", "SOL"), "D": ("B0p", "BTC"),
           "C": ("C", "BTC"), "B0": ("B0", "BTC"), "B1": ("B1", "BTC"), "B2": ("B2", "BTC")}
F_WINDOW, F_TARGET = 40, {"F1": 0.56, "F2": 0.60}
BETA_GRID = [round(0.025 * k, 3) for k in range(7)]       # 0, 0.025, ..., 0.150
CALIB_SEEDS = range(2000, 2020)
CHUNK = 10                                                 # Layer 2 seeds per part file
REPL_COUNTS = (2008, 2647, 11034, 8049, 5809)              # E15 A-BTC seeds 0-4, S3 PBO counts of 12,870
E14_DATA_SHA = {"BTC": "19dd18c523032b82723e07f45410f38f9158136f7f8c48036761dade79c44d29",
                "ETH": "8d0851f3b7e21a250dbc8a88d13f6d5c96959e61a8bb5e0469f497027ac0d5fe",
                "SOL": "33db1e362fd1ae6e51b03950e983dda1c240fe3f37b083f327c6284e21d6af95"}
DEV_PREFIX, DEV_RS, DEV_BARS = 99, 99_000_000, {"BTC": 3133, "ETH": 2281, "SOL": 2058}
CATBOOST_E15 = "1.2.10"                                    # E14 / E15 run logs (results/inbox7/summary.json)
VF_COMMIT = "018f4ea26dc2a1a1ec3bf38d4a2dcb73aa659861"    # valid-framework worktree of E14 / E15
PREREG_REL = "experiments/cscv_calibration/PREREGISTRATION.md"
KST = timezone(timedelta(hours=9))
BETA_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) β (F1|F2) = (\d\.\d{3}) \(interpolated (\d\.\d{5}); "
                     r"target mean holdout AUC (\d\.\d{2}); seeds 2000–2019\)$")
DROP_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) β (F1|F2) dropped \(no bracketing pair up to 0\.150\)$")

IMPLEMENTATION_NOTES = [
    "Equicorrelated Gaussian rows: Z = rng.standard_normal((n, M + 1)); g = sqrt(rho) * Z[:, :1] + sqrt(1 - rho) "
    "* Z[:, 1:] (one common factor; unit variance, Pearson correlation rho); also the Gaussian copula of GARCH",
    "IID: e = g(T rows). MA10: g drawn for T + 9 rows (rows 0-8 are the lags before t = 0); e(t) = (sum over "
    "k = 0..9 of xi(t - k), added in order k = 0..9) / sqrt(10)",
    "GARCH draw order: eta = rng.standard_t(5, T + 500) / sqrt(5/3) first, then Z for the T kept periods only "
    "(h depends on eta alone, so no column shocks are drawn for burn-in periods); h(0) = 1, u = sqrt(h(t-1)) * "
    "eta(t-1), h(t) = 0.02 + 0.08 * u * u + 0.90 * h(t-1) in Python floats; periods 0-499 dropped; e = sqrt(h) * eps",
    "t(5) copula marginal: eps = F_t5^-1(Phi(g)) / sqrt(5/3), evaluated sign-symmetrically as "
    "-sign(g) * stdtrit(5, ndtr(-|g|)) so that the upper tail is not rounded to Phi = 1",
    "rho_P(rho): nested scipy.integrate.quad of E[eps_1 eps_2] over +-12 sd (epsrel 1e-12); recorded in every "
    "run log; the GARCH correlation test targets the integrated value",
    "Means: mu_m = SR_m / sqrt(252) * sigma_m with SR_m the arm's annual Sharpe ratio; N2: SR_m = -1.0 * (m - 1) "
    "/ (M - 1); N3: mu = mu(P2) - mu_1(P2) for every column, r = mu + sigma * e with e from the P2 seed [5, c, j]",
    "Unequal volatility (cell 216): sigma_m = 0.005 * 4 ** ((m - 1) / (M - 1)), reference-cell T, M, S, rho, "
    "serial; arms N1, N2, P2; seeds [a, 216, j]",
    "Full-sample Sharpe ratio (best configuration, reported per series): mean / (sd + 1e-10), sd with divisor "
    "n, ties to the first configuration (np.argmax); Layer 1 reports the arm's population annual SR of it",
    "PBO is stored as integer counts k of K splits (PBO = k / K, bitwise equal to pbo_fast, asserted at every "
    "call); last-listed and mean-rank rules and the count of splits with an in-sample tie come from "
    "tie_rules(), unit-tested equal to the per-split rule of experiments/inbox7/ties.py",
    "PBO screens: PBO < tau (strict), tau in {0.05, 0.10, 0.20, 0.50}; computed in summarize.py from the counts",
    "Stationary bootstrap: rng = default_rng(stream); start = rng.integers(T); u = rng.random(T - 1); jump = "
    "rng.integers(0, T, size=T - 1) (the order of E15 make_sb_series, on a Generator); a block starts at t if "
    "u[t-1] < 1/L (index jump[t-1]), else index = previous + 1 mod T; vectorised, unit-tested equal to the loop",
    "Calibrated p: D = demeaned(R) = R - R.mean(0) (checked |column mean| <= 1e-12); boot(D, stream, Ls) returns "
    "the indices and D[idx] per L (not demeaned again); PBO*_b = pbo_fast(cscv_matrices(D[idx_b], S)); PBO_obs = "
    "pbo_fast on R (first-listed tie rule); p = (1 + #{PBO*_b <= PBO_obs}) / (B + 1); the section 6 demeaning and "
    "no-re-centring tests call demeaned() and boot()",
    "L sensitivity (L = 5, 50) reuses the streams of L = 20 (same start, u, jump); only the block-start rule "
    "u < 1/L changes; the Reality Check is computed at L = 20 only",
    "Reality Check: V_obs = max_m sqrt(T) * mean_m; V*_b = max_m sqrt(T) * (mean(R[idx_b])_m - mean_m) with the "
    "L = 20 indices of the same stream; p_RC = (1 + #{V*_b >= V_obs}) / (B + 1)",
    "Stage 2 regenerates series j = 0..499 from their Stage 1 seeds; summarize.py stops if an observed PBO "
    "differs from Stage 1's",
    "Layer 2 grid order: [(d, lr, cw, l2) for d in (4, 6, 8) for lr in (0.03, 0.1) for cw in (None, "
    "'Balanced') for l2 in (3, 10)]; sub-grids keep this order (M = 8 equals E15 GRID_P order, unit-tested; "
    "M = 16 = columns 0-15)",
    "CatBoost: CatBoostClassifier(depth, iterations=100, learning_rate, verbose=0, random_seed=42, l2_leaf_reg, "
    "auto_class_weights only when set); thread_count not set (CatBoost default -1 = all cores), as E15",
    "Layer 2 arms via E15 build(): A-<asset> = 'A', C = 'C', B0/B1/B2 = same names, D = 'B0p' (E14 base, not "
    "demeaned: identical to e14.make_series(seed, r, ...)); F1/F2 = make_series_w(window=40) on demean(r); R as "
    "E14 cmd_real (ret = close.pct_change().fillna(0) on the actual BTC series)",
    "F calibration: seeds whose prepare() returns None are skipped (E14 calib convention); bracketing pair = "
    "first i from beta = 0 up with (AUC_i - target) * (AUC_i+1 - target) <= 0; beta = beta_i + (target - AUC_i) "
    "* (beta_i+1 - beta_i) / (AUC_i+1 - AUC_i) (beta_i if AUC_i = AUC_i+1); the run uses beta rounded to 3 "
    "decimals (E14 convention) and stores the unrounded value beside it; an arm without a bracketing pair is "
    "stored as {dropped: true, reason}",
    "Change-log beta lines are generated by 'betalog' (date in KST), one per arm: '- YYYY-MM-DD β F1 = 0.066 "
    "(interpolated 0.06573; target mean holdout AUC 0.56; seeds 2000–2019)' or '- YYYY-MM-DD β F2 dropped (no "
    "bracketing pair up to 0.150)'; '--append' replaces the '(empty)' placeholder only if it is still there",
    "Any registered Stage 3 unit with seeds 1000-1099 (B0, B1, B2, F1, F2) refuses unless (i) the Change log has "
    "exactly one strictly parsed beta line per arm, equal to beta_F.json (beta at 3 decimals, interpolated value "
    "at 5, target), (ii) git diff --quiet HEAD -- PREREGISTRATION.md, (iii) after git fetch origin, the last "
    "commit touching PREREGISTRATION.md is an ancestor of origin/main; both shas go to the run log; dropped arms "
    "are skipped and reported as dropped",
    "Registered Layer 2 commands (repl, calib, stage3) refuse a CatBoost version other than E15's 1.2.10",
    "Layer 2 bootstrap streams [8, k, s, b] are shared by (M = 8, S = 16), (M = 16, S = 8) and the L sensitivity",
    "Median pairwise correlation: np.median of the upper triangle of np.corrcoef of the M columns (NaN if a "
    "column is constant)",
    "Layer 2 invalid series (prepare() is None) are rows with valid = 0; summarize.py counts them as not passing",
    "Dev namespace: seeds [99, ...] in Layer 1 and Layer 2 streams, Layer 2 RandomState seed 99,000,000 + s, "
    "synthetic data_sources 'synthetic' with the real lengths; outputs under results/ are refused",
    "Timing commands use dev-namespace seeds on the run's data (no registered series is run); Stage 1 times "
    "every (T, M, S, serial) shape at rho = 0.5, each arm; Stages 1-2 project with 2 workers halving the wall "
    "time (one BLAS thread each, scaling measured); Stage 3 records the halved and the unhalved projection and "
    "decides B on the unhalved one, because CatBoost uses all cores in each worker",
    "Checks before Stage 1 (prereg section 6): every registered stage refuses unless checks/unit.json and "
    "checks/repl.json pass and checks/timing_s1.json, timing_s2.json and timing_s3.json exist, all written in the "
    "registered namespace by this run.py (same sha256), and Stage 1's timing is within its limit",
    "Stage lock: checks/stage<k>_lock.json freezes B, the timing file's sha256 and run.py's sha256 when a shard "
    "of stage k starts; a later shard refuses if any differs once parts of that stage exist. Stage 1's lock also "
    "freezes the sha256 of timing_s2.json and timing_s3.json; Stages 2 and 3 refuse to start before Stage 1 has "
    "started or with a timing file other than the frozen one, so their B is fixed before any Stage 1 result exists",
    "Layer 1 (Stages 1-2 and their timing) runs numpy BLAS with one thread (threadpoolctl); unit-tested to give "
    "bitwise-equal PBO and calibrated-p outputs; Layer 2 keeps the BLAS and CatBoost thread defaults of E15",
    "Parts are written atomically per (cell, arm) in Stages 1-2 and per 10-seed chunk in Stage 3 (temporary "
    "names carry the pid); existing parts are skipped (resume); each shard writes its own run log",
    "Shard lock: locks/<stage, shard, filters>.lock created with O_CREAT | O_EXCL, holding pid and host; a second "
    "launch of the same shard refuses; a lock of a dead pid on the same host is cleared; removed at exit",
    "Namespace marker: OUT/NAMESPACE holds 'reg' or 'dev'; a run of the other namespace (or into a directory "
    "whose parts carry the other namespace) refuses",
    "Safety checks (data and prereg digests, VF worktree at 018f4ea, namespaces, dev-only options, seed ranges, "
    "unit invariants) raise SystemExit through need(), so they hold under python -O",
    "Unit-test seeds: default_rng([4242 + i, ...]), never a registered tuple; generator tests at M = 8",
]


# ---------------------------------------------------------------- namespaces, seeds, data
def gen(dev, *key):
    return np.random.default_rng(([DEV_PREFIX] if dev else []) + list(key))


def stream(dev, *key):
    return ([DEV_PREFIX] if dev else []) + list(key)


def rs_seed(dev, s):
    return DEV_RS + s if dev else s


_DATA = {}


def returns(asset, syn):
    """(df, daily returns) as E14 asset_returns; syn: synthetic data of the same length (dev only)."""
    if (asset, syn) not in _DATA:
        if syn:
            df = e14.load_ohlcv("synthetic", asset, n_bars=DEV_BARS[asset])
            df.attrs = {}   # the synthetic regime array in attrs breaks pandas concat in prepare (dev only)
            _DATA[asset, syn] = (df, df["close"].pct_change().dropna().values)
        else:
            _DATA[asset, syn] = e14.asset_returns(asset)
    return _DATA[asset, syn]


def cache(syn):
    return {a: returns(a, syn) for a in ("BTC", "ETH", "SOL")}


def need(cond, msg):
    """Explicit check (survives python -O): stops the command with a message."""
    if not cond:
        raise SystemExit(f"STOP: {msg}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def data_digests(dev):
    if dev:
        return {"source": "synthetic (dev)"}
    out = {a: sha256(data_sources.real_data_path(a)) for a in E14_DATA_SHA}
    bad = [a for a in out if out[a] != E14_DATA_SHA[a]]
    need(not bad, f"data sha256 differs from E14 for {bad}")
    return out


def run_sha():
    return sha256(os.path.abspath(__file__))


def vf_head():
    h = subprocess.run(["git", "-C", e14.VF, "rev-parse", "HEAD"], capture_output=True, text=True,
                       timeout=30).stdout.strip()
    need(h == VF_COMMIT, f"VF worktree {e14.VF} is at {h or '(not a git checkout)'}, not {VF_COMMIT}")
    return {"path": e14.VF, "head": h}


def prereg_check():
    t = open(PREREG, encoding="utf-8").read()
    i = t.index("## Change log")
    body = hashlib.sha256(t[:i].encode()).hexdigest()
    need(body == PREREG_BODY_SHA, "PREREGISTRATION.md (before the Change log) differs from 2ead385")
    return {"file_sha256": sha256(PREREG), "body_sha256": body, "change_log": t[i:]}


# ---------------------------------------------------------------- Layer 1 generator
def equi(z, rho):
    return math.sqrt(rho) * z[:, :1] + math.sqrt(1 - rho) * z[:, 1:]


def garch_h(eta):
    e = eta.tolist()
    h = [1.0] * len(e)
    for t in range(1, len(e)):
        u = math.sqrt(h[t - 1]) * e[t - 1]
        h[t] = 0.02 + 0.08 * u * u + 0.90 * h[t - 1]
    return np.array(h)


def t5(g):
    q = stdtrit(5, ndtr(-np.abs(g)))
    return np.where(g > 0, -q, q) / T5_SD


def shocks(rng, T, M, rho, serial):
    """e (T x M): unit variance; Pearson correlation rho (IID, MA10) or rho_P(rho) (GARCH)."""
    if serial == "IID":
        return equi(rng.standard_normal((T, M + 1)), rho)
    if serial == "MA10":
        xi = equi(rng.standard_normal((T + 9, M + 1)), rho)
        e = xi[9:].copy()
        for k in range(1, 10):
            e += xi[9 - k:9 - k + T]
        e /= math.sqrt(10)
        return e
    eta = rng.standard_t(5, size=T + BURN) / T5_SD
    h = garch_h(eta)[BURN:]
    return np.sqrt(h)[:, None] * t5(equi(rng.standard_normal((T, M + 1)), rho))


_RHO_P = {}


def rho_p(rho, lim=12.0):
    """Pearson correlation of two standardised t(5) marginals joined by a Gaussian copula with correlation rho."""
    if rho not in _RHO_P:
        s = math.sqrt(1 - rho * rho)
        phi = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)  # noqa: E731
        g1 = lambda x: float(t5(np.array(x)))  # noqa: E731
        inner = lambda x: integrate.quad(lambda z: g1(rho * x + s * z) * phi(z), -lim, lim,  # noqa: E731
                                         epsabs=1e-13, epsrel=1e-12, limit=400)[0]
        _RHO_P[rho] = 0.0 if rho == 0 else integrate.quad(lambda x: g1(x) * phi(x) * inner(x), -lim, lim,
                                                          epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return _RHO_P[rho]


def ann_sr(arm, M):
    m = np.arange(M)
    if arm == "N1":
        return np.zeros(M)
    if arm == "N2":
        return -1.0 * m / (M - 1)
    if arm == "N3":
        return np.where(m == 0, 0.0, -EDGE["P2"])
    return np.where(m == 0, EDGE[arm], 0.0)


def sigmas(M, uneq=False):
    return 0.005 * 4.0 ** (np.arange(M) / (M - 1)) if uneq else np.full(M, SIGMA)


def means(arm, sig):
    if arm == "N3":
        mp = ann_sr("P2", len(sig)) / math.sqrt(ANN) * sig
        return mp - mp[0]
    return ann_sr(arm, len(sig)) / math.sqrt(ANN) * sig


def l1_series(arm, c, j, dev):
    """Series (arm, c, j) of Layer 1: T x M returns; N3 uses the shocks of P2 series j."""
    T, M, S, rho, serial = CELLS[REF if c == UNEQ else c]
    sig = sigmas(M, c == UNEQ)
    e = shocks(gen(dev, ARM1["P2" if arm == "N3" else arm], c, j), T, M, rho, serial)
    return means(arm, sig) + sig * e


# ---------------------------------------------------------------- statistics
def tie_rules(is_m, oos_m):
    """Counts of splits with logit(omega_bar) <= 0 under the first-listed, last-listed and mean-rank rules of
    experiments/inbox7/ties.py, and the count of splits with an in-sample tie (CSCV matrices, no NaN)."""
    M, K = is_m.shape
    tied = is_m == is_m.max(0)
    nt = tied.sum(0)
    cols = np.arange(K)

    def rank(i):  # average rank (scipy rankdata 'average') of oos_m[i[k], k] within column k
        v = oos_m[i, cols]
        return (oos_m < v).sum(0) + ((oos_m == v).sum(0) + 1) / 2

    r = {"first": rank(tied.argmax(0)), "last": rank(M - 1 - tied[::-1].argmax(0))}
    r["mean_rank"] = r["first"].copy()
    sub = np.flatnonzero(nt > 1)
    if sub.size:
        r["mean_rank"][sub] = (rankdata(oos_m[:, sub], axis=0) * tied[:, sub]).sum(0) / nt[sub]
    out = {}
    for k, v in r.items():
        w = v / (M + 1)
        out[k] = int(np.sum(np.log(w / (1 - w)) <= 0))
    out["tie"] = int(np.sum(nt > 1))
    return out


def pbo_stats(R, S):
    """E15 canonical CSCV (unchanged) + the other tie rules; PBO = k_first / K."""
    is_m, oos_m = e15.cscv_matrices(R, S)
    K = is_m.shape[1]
    p = e15.pbo_fast(is_m, oos_m)
    t = tie_rules(is_m, oos_m)
    need(t["first"] / K == p, "tie_rules first-listed count != pbo_fast")
    return {"K": K, "k_first": t["first"], "k_last": t["last"], "k_mean": t["mean_rank"], "k_tie": t["tie"]}


def best_full(R):
    sr = R.mean(0) / (R.std(0) + EPS)
    b = int(np.argmax(sr))
    return b, float(sr[b])


def med_corr(R):
    c = np.corrcoef(R, rowvar=False)
    return float(np.median(c[np.triu_indices(R.shape[1], 1)]))


def sb_draw(rng, n):
    return rng.integers(n), rng.random(n - 1), rng.integers(0, n, size=n - 1)


def sb_index(start, u, jump, n, L):
    new = np.concatenate(([True], u < 1.0 / L))
    val = np.concatenate(([start], jump))
    pos = np.arange(n)
    s = np.maximum.accumulate(np.where(new, pos, 0))
    return (val[s] + (pos - s)) % n


def sb_loop(start, u, jump, n, L):
    """E15 make_sb_series index rule, written as its loop (unit test reference)."""
    idx = np.empty(n, dtype=int)
    idx[0] = start
    for t in range(1, n):
        idx[t] = jump[t - 1] if u[t - 1] < 1.0 / L else (idx[t - 1] + 1) % n
    return idx


def demeaned(R):
    """Step 1 of the calibrated p-value: each column minus its sample mean (checked to 1e-12)."""
    D = R - R.mean(0)
    dm = float(np.abs(D.mean(0)).max())
    need(dm <= 1e-12, f"demeaned column mean {dm}")
    return D


def boot(D, sd, Ls):
    """Step 2, one resample: stream sd -> indices per L and D[idx] per L (not demeaned again)."""
    T = D.shape[0]
    start, u, jump = sb_draw(np.random.default_rng(sd), T)
    idx = {L: sb_index(start, u, jump, T, L) for L in Ls}
    return idx, {L: D[i] for L, i in idx.items()}


def calibrated(R, S, streams, Ls=(L0,)):
    """Calibrated p-value per L and the Reality Check at L0, on the same index streams (prereg section 3)."""
    need(L0 in Ls, "L0 must be among the block lengths")
    T = R.shape[0]
    B = len(streams)
    D = demeaned(R)
    is_m, oos_m = e15.cscv_matrices(R, S)
    K = is_m.shape[1]
    p_obs = e15.pbo_fast(is_m, oos_m)
    mean = R.mean(0)
    v_obs = float((math.sqrt(T) * mean).max())
    star = {L: np.empty(B) for L in Ls}
    v_star = np.empty(B)
    for b, sd in enumerate(streams):
        idx, Db = boot(D, sd, Ls)
        for L in Ls:
            star[L][b] = e15.pbo_fast(*e15.cscv_matrices(Db[L], S))
        v_star[b] = (math.sqrt(T) * (R[idx[L0]].mean(0) - mean)).max()
    out = {"K": K, "k_obs": int(round(p_obs * K)), "v_obs": v_obs}
    need(out["k_obs"] / K == p_obs, "observed PBO is not k/K")
    counts = {}
    for L in Ls:
        le = int(np.sum(star[L] <= p_obs))
        p = (1 + le) / (B + 1)
        need(1 / (B + 1) <= p <= 1, f"p = {p} outside [1/(B+1), 1]")
        out[f"le_L{L}"], out[f"p_L{L}"] = le, p
        counts[L] = np.rint(star[L] * K).astype(np.uint16)
        need(np.array_equal(counts[L] / K, star[L]), "PBO* is not k/K")
    out["ge_rc"] = int(np.sum(v_star >= v_obs))
    out["p_rc"] = (1 + out["ge_rc"]) / (B + 1)
    return out, counts


# ---------------------------------------------------------------- Layer 2
def make_c(cfg):
    from catboost import CatBoostClassifier
    d, lr, cw, l2 = cfg
    kw = {"auto_class_weights": cw} if cw else {}
    return CatBoostClassifier(depth=d, iterations=100, learning_rate=lr, verbose=0, random_seed=42,
                              l2_leaf_reg=l2, **kw)


def make_series_w(seed, ret_actual, n_days, ref_index, beta=None, window=5):
    """e14.make_series with the planted sum over `window` days (window 5 = E14 bitwise; 40 = F1, F2)."""
    rng = np.random.RandomState(seed)
    r = rng.choice(ret_actual, size=n_days, replace=True)
    if beta is not None:
        mu, sig = r.mean(), r.std()
        rp = r.copy()
        for t in range(n_days - 1):
            z = 0.0 if t < window - 1 else (rp[t - window + 1:t + 1].sum() - window * mu) / (sig * np.sqrt(window))
            rp[t + 1] = r[t + 1] + beta * sig * z
        r = rp
    price = 10000 * np.exp(np.cumsum(r))
    df = pd.DataFrame({
        "open": price,
        "high": price * (1 + np.abs(rng.normal(0, 0.01, n_days))),
        "low": price * (1 - np.abs(rng.normal(0, 0.01, n_days))),
        "close": price,
        "volume": rng.lognormal(20, 1, n_days),
    }, index=ref_index[:n_days])
    return df, r


def l2_build(arm, s, dev, beta=None, syn=None):
    """Layer 2 series of an arm; dev = seed namespace, syn = synthetic data (default: syn = dev)."""
    syn = dev if syn is None else syn
    if arm == "R":
        df, _ = returns("BTC", syn)
        ret = df["close"].pct_change().fillna(0).values
        return e14.prepare(df[["open", "high", "low", "close", "volume"]].copy(), ret)
    if arm in F_TARGET:
        dfa, r = returns("BTC", syn)
        df, ret = make_series_w(rs_seed(dev, s), e15.demean(r), len(r), dfa.index, beta, F_WINDOW)
        return e14.prepare(df, ret)
    a, asset = E15_ARM[arm]
    return e15.build(a, asset, rs_seed(dev, s), cache(syn))


def l2_eval(arm, s, dev, B, beta=None, syn=None):
    row, arrs = {"ns": "dev" if dev else "reg", "arm": arm, "seed": s}, {}
    t0 = time.time()
    p = l2_build(arm, s, dev, beta, syn)
    if p is None:
        return {**row, "valid": 0}, arrs
    X, y, fwd, split = p
    R = e15.cv6_returns(X, fwd, y, GRID24, make_c)
    row.update({"valid": 1, "events": len(X)})
    arrs[f"{arm}_{s}_R"] = R
    for M, cols in SUB.items():
        Rm = R[:, cols]
        b, sr = best_full(Rm)
        row.update({f"M{M}_best": cols[b], f"M{M}_best_sr_pe": sr, f"M{M}_med_corr": med_corr(Rm)})
        for S in SS:
            row.update({f"M{M}_S{S}_{k}": v for k, v in pbo_stats(Rm, S).items()})
    t1 = time.time()
    row["t_fit"] = t1 - t0   # 24 x 6 fits + per-sub-grid statistics; t_boot below scales with B
    for M, S in CALIB_MS:
        Ls = (L0,) + L_SENS if (arm == "A-BTC" and (M, S) == (8, 16)) else (L0,)
        res, cnt = calibrated(R[:, SUB[M]], S, [stream(dev, 8, ARM2[arm], s, b) for b in range(B)], Ls)
        row.update({f"cal_M{M}_S{S}_{k}": v for k, v in res.items()})
        arrs.update({f"{arm}_{s}_M{M}_S{S}_L{L}_star": v for L, v in cnt.items()})
    row.update({"B": B, "t_boot": time.time() - t1, "t_total": time.time() - t0})
    return row, arrs


# ---------------------------------------------------------------- work units
def s1_unit(c, arm, dev, n=None):
    T, M, S, rho, serial = CELLS[REF if c == UNEQ else c]
    sr = ann_sr(arm, M)
    rows = []
    for j in range(n or (UNEQ_REPS if c == UNEQ else REPS1)[arm]):
        R = l1_series(arm, c, j, dev)
        b, _ = best_full(R)
        rows.append({"ns": "dev" if dev else "reg", "cell": c, "arm": arm, "j": j, **pbo_stats(R, S),
                     "best": b, "best_pop_sr": float(sr[b])})
    return rows, None


def s2_unit(c, arm, dev, B, n=None):
    S = CELLS[c][2]
    a = ARM1["P2" if arm == "N3" else arm]   # N3 uses the stream of its paired P2 series
    Ls = (L0,) + L_SENS if (c == REF and arm in ("N1", "P2")) else (L0,)
    rows, stars = [], {L: [] for L in Ls}
    for j in range(n or S2_REPS):
        res, cnt = calibrated(l1_series(arm, c, j, dev), S, [stream(dev, 7, a, c, j, b) for b in range(B)], Ls)
        rows.append({"ns": "dev" if dev else "reg", "cell": c, "arm": arm, "j": j, "B": B, **res})
        for L in Ls:
            stars[L].append(cnt[L])
    return rows, {f"pbo_star_counts_L{L}": np.stack(v) for L, v in stars.items()}


def s3_unit(arm, seeds, dev, B, beta):
    rows, arrs = [], {}
    for s in seeds:
        row, a = l2_eval(arm, s, dev, B, beta)
        rows.append(row)
        arrs.update(a)
    return rows, arrs


def units_of(stage, a):
    if stage == 1:
        cells = range(*rng_of(a.cells, 0, UNEQ)) if a.cells else range(UNEQ + 1)
        arms = a.arms.split(",") if a.arms else list(ARM1)
        us = [(c, arm) for c in cells for arm in arms if arm in (UNEQ_REPS if c == UNEQ else REPS1)]
    elif stage == 2:
        cells = [c for c in S2_CELLS if not a.cells or c in range(*rng_of(a.cells, 0, UNEQ))]
        us = [(c, arm) for c in cells for arm in (a.arms.split(",") if a.arms else S2_ARMS) if arm in S2_ARMS]
    else:
        arms = a.arms.split(",") if a.arms else list(ARM2)
        us = []
        keep = range(*rng_of(a.seeds, 0, 10 ** 4)) if a.seeds else None
        for arm in arms:
            sd = list(SEEDS2[arm])
            chunks = [tuple(sd[i:i + CHUNK]) for i in range(0, len(sd), CHUNK)]   # fixed chunk boundaries
            if keep is not None:
                chunks = [ch for ch in chunks if set(ch) & set(keep)]
                need(all(set(ch) <= set(keep) for ch in chunks), f"--seeds must cover whole chunks of {CHUNK}")
            us += [(arm, ch) for ch in chunks]
    k, n = map(int, a.shard.split("/"))
    return us[k::n]


def rng_of(s, lo, hi):
    x, y = s.split("-")
    x, y = int(x), int(y)
    need(lo <= x <= y <= hi, f"range {s} outside {lo}-{hi}")
    return x, y + 1


def part_path(out, stage, u):
    if stage in (1, 2):
        return os.path.join(out, "parts", f"s{stage}", f"c{u[0]:03d}_{u[1]}.csv")
    return os.path.join(out, "parts", "s3", f"{u[0]}_{u[1][0]:04d}-{u[1][-1]:04d}.csv")


def save_part(path, rows, arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pid = os.getpid()
    if arrs is not None:
        tmp = f"{path[:-4]}.{pid}.tmp.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, path[:-4] + ".npz")
    pd.DataFrame(rows).to_csv(f"{path}.{pid}.tmp", index=False)
    os.replace(f"{path}.{pid}.tmp", path)


# ---------------------------------------------------------------- run log, checks
STOP_CONDITIONS = [
    "a check before Stage 1 is missing or did not pass (checks/unit.json, repl.json, timing_s1/2/3.json; same "
    "run.py sha256; registered namespace): every stage refuses to start",
    "Stage 1 timing projects more than 10 h with 2 workers: the stages refuse to start (prereg section 6)",
    "Stage 2 timing over 10 h (2 workers) / Stage 3 timing over 12 h (unhalved) at B = 199: B = 99 in that stage",
    "the stage lock (B, timing sha256, run.py sha256) differs once parts of the stage exist, or Stage 2/3 timing "
    "differs from the file frozen when Stage 1 started: refuse",
    "PREREGISTRATION.md before the Change log differs from 2ead385, the data sha256 differs from E14, or the VF "
    "worktree is not at 018f4ea: refuse",
    "seeds 1000-1099 before both betas are in the Change log, committed and on origin/main: refuse",
    "the same shard is already running (shard lock), or the output directory holds the other namespace: refuse",
    "a check fails inside a unit (demeaned column mean > 1e-12, p outside [1/(B+1), 1], tie_rules != pbo_fast): "
    "the shard stops; finished part files are kept and skipped on restart",
]


def blas_one(stage):
    """Layer 1 (Stages 1-2) runs numpy BLAS single-threaded: two workers would otherwise oversubscribe the cores.
    Results are unchanged (unit test blas_threads_invariant). Layer 2 keeps E15's defaults."""
    return threadpool_limits(limits=1, user_api="blas") if stage in (1, 2) else contextlib.nullcontext()


def blas_one_info(stage):
    with blas_one(stage):
        return blas_info()


def blas_info():
    return [{k: d.get(k) for k in ("internal_api", "version", "num_threads")} for d in threadpool_info()
            if d.get("user_api") == "blas"]


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    for m in ("numpy", "scipy", "pandas", "sklearn", "catboost", "lightgbm"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    return out


def catboost_threads():
    """CatBoost has no resolved thread_count attribute; record the default and the native threads seen in a fit."""
    out = {"thread_count_param": "not set (CatBoost default -1 = all processor cores)", "os_cpu_count": os.cpu_count()}
    try:
        import psutil
    except ImportError:
        return {**out, "threads_during_fit": None}
    proc, peak, stop = psutil.Process(), [0], threading.Event()
    base = proc.num_threads()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.num_threads())
            time.sleep(0.002)

    th = threading.Thread(target=poll)
    th.start()
    rng = np.random.default_rng([4299])
    X = rng.standard_normal((4000, 20))
    make_c((8, 0.1, None, 3)).fit(X, (X[:, 0] + rng.standard_normal(4000) > 0).astype(int))
    stop.set()
    th.join()
    return {**out, "process_threads_before_fit": base, "process_threads_peak_during_fit": peak[0]}


def git(*args, cwd=ROOT):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=300)


def git_head():
    try:
        return git("rev-parse", "HEAD").stdout.strip() or None
    except Exception:
        return None


def ns_name(dev):
    return "dev" if dev else "reg"


def check_out(out, dev):
    """Output directory: dev never under results/; a NAMESPACE marker keeps dev and registered outputs apart."""
    real = os.path.realpath(out)
    if dev:
        need(not real.startswith(os.path.realpath(os.path.join(ROOT, "results"))), "--dev output under results/")
    os.makedirs(out, exist_ok=True)
    mk = os.path.join(out, "NAMESPACE")
    if os.path.exists(mk):
        have = open(mk).read().strip()
        need(have == ns_name(dev), f"{out} holds the '{have}' namespace; this run is '{ns_name(dev)}'")
        return
    for f in sorted(glob.glob(os.path.join(out, "parts", "s*", "*.csv")))[:3]:
        have = pd.read_csv(f, nrows=1)["ns"].iloc[0]
        need(have == ns_name(dev), f"{f} holds the '{have}' namespace; this run is '{ns_name(dev)}'")
    with open(f"{mk}.{os.getpid()}.tmp", "w") as f:
        f.write(ns_name(dev) + "\n")
    os.replace(f"{mk}.{os.getpid()}.tmp", mk)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def check_stamp(dev):
    """Fields that tie a check file to this run.py and namespace."""
    return {"ns": ns_name(dev), "run_py_sha256": run_sha()}


def require_checks(out, stage):
    """Prereg section 6: unit tests, the replication and the timing of all three stages come before Stage 1."""
    ck = os.path.join(out, "checks")
    got = {}
    for f in ("unit.json", "repl.json", "timing_s1.json", "timing_s2.json", "timing_s3.json"):
        p = os.path.join(ck, f)
        need(os.path.exists(p), f"{p} missing (prereg section 6: run unit, repl and time --stage 1, 2 and 3 first)")
        d = read_json(p)
        need(d.get("ns") == "reg", f"{p} was not written in the registered namespace")
        need(d.get("run_py_sha256") == run_sha(), f"{p} was written by another version of run.py")
        need(f not in ("unit.json", "repl.json") or d.get("pass") is True, f"{p} did not pass")
        got[f] = d
    need(got["timing_s1.json"]["within_limit"], "Stage 1 timing over the limit (prereg section 6: the run stops)")
    return got[f"timing_s{stage}.json"], os.path.join(ck, f"timing_s{stage}.json")


def stage_lock(out, stage, B, timing_path):
    """Freeze B, the timing file and run.py for a stage; refuse a change once parts of the stage exist.
    Stage 1's lock also freezes the timing files of Stages 2 and 3, so their B cannot be re-decided after
    Stage 1 results exist; Stages 2 and 3 start only after Stage 1 and must use those frozen files."""
    want = {"stage": stage, "B": B, "run_py_sha256": run_sha(),
            "timing_sha256": sha256(timing_path) if timing_path and os.path.exists(timing_path) else None}
    ck = os.path.join(out, "checks")
    if stage == 1:
        for k in (2, 3):
            tk = os.path.join(ck, f"timing_s{k}.json")
            want[f"timing_s{k}_sha256"] = sha256(tk) if os.path.exists(tk) else None
    else:
        p1 = os.path.join(ck, "stage1_lock.json")
        need(os.path.exists(p1), f"{p1} missing: Stage {stage} starts only after Stage 1 has started")
        need(read_json(p1).get(f"timing_s{stage}_sha256") == want["timing_sha256"],
             f"timing_s{stage}.json differs from the one frozen when Stage 1 started ({p1})")
    p = os.path.join(out, "checks", f"stage{stage}_lock.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and read_json(p) != want:
        need(not glob.glob(os.path.join(out, "parts", f"s{stage}", "*.csv")),
             f"{p} freezes {read_json(p)}; this run would use {want}")
        os.remove(p)   # no part was written under the old lock
    if not os.path.exists(p):
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(want, f, indent=1)
        try:
            os.link(tmp, p)   # atomic: the first shard writes the lock
        except FileExistsError:
            pass
        finally:
            os.remove(tmp)
    need(read_json(p) == want, f"{p} freezes {read_json(p)}; this run would use {want}")
    return want


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, TypeError, ValueError):
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def shard_lock(out, stage, a):
    """One process per shard: locks/<stage, shard, filters>.lock with O_CREAT | O_EXCL."""
    key = re.sub(r"[^A-Za-z0-9_.-]", "-", f"stage{stage}_{a.shard}_c{a.cells}_a{a.arms}_s{a.seeds}")
    p = os.path.join(out, "locks", key + ".lock")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    me = {"pid": os.getpid(), "host": platform.node(), "start": datetime.now(timezone.utc).isoformat()}
    for attempt in (0, 1):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                other = read_json(p)
            except Exception:
                other = {}
            stale = other.get("host") == me["host"] and not pid_alive(other.get("pid"))
            need(attempt == 0 and stale, f"{p} is held by {other or '(unreadable)'}: this shard is already running "
                                         "(delete the lock only if that process is gone)")
            os.remove(p)
    with os.fdopen(fd, "w") as f:
        json.dump(me, f)
    try:
        yield p
    finally:
        if os.path.exists(p):
            os.remove(p)


def require_catboost():
    import catboost
    need(catboost.__version__ == CATBOOST_E15, f"catboost {catboost.__version__}; the prereg fixes E15's {CATBOOST_E15}")


# ---------------------------------------------------------------- beta publication (prereg section 2)
def read_beta(out, dev):
    p = os.path.join(out, "beta_F.json")
    need(os.path.exists(p), f"{p} missing (run.py calib first)")
    d = read_json(p)
    need(d.get("ns") == ns_name(dev), f"{p} namespace {d.get('ns')} differs from '{ns_name(dev)}'")
    for arm in F_TARGET:
        need(isinstance(d.get(arm), dict), f"{p}: {arm} entry missing")
    return d, p


def beta_lines(d, date):
    out = []
    for arm, tgt in F_TARGET.items():
        v = d[arm]
        if v.get("dropped"):
            out.append(f"- {date} β {arm} dropped (no bracketing pair up to 0.150)")
        else:
            out.append(f"- {date} β {arm} = {v['beta']:.3f} (interpolated {v['beta_exact']:.5f}; "
                       f"target mean holdout AUC {tgt:.2f}; seeds 2000–2019)")
    return out


def parse_beta_log(change_log, d):
    """Strict parse: exactly one beta line per arm in the Change log, equal to beta_F.json."""
    got = {}
    for line in change_log.splitlines():
        if not re.search(r"β (F1|F2)\b", line):
            continue
        m, k = BETA_RE.match(line), DROP_RE.match(line)
        need(m or k, f"malformed beta line in the prereg Change log: {line!r}")
        arm = (m or k).group(2)
        need(arm not in got, f"more than one beta line for {arm} in the prereg Change log")
        got[arm] = {"beta": m.group(3), "exact": m.group(4), "target": m.group(5)} if m else {"dropped": True}
    for arm, tgt in F_TARGET.items():
        need(arm in got, f"no beta line for {arm} in the prereg Change log")
        v = d[arm]
        want = {"dropped": True} if v.get("dropped") else \
            {"beta": f"{v['beta']:.3f}", "exact": f"{v['beta_exact']:.5f}", "target": f"{tgt:.2f}"}
        need(got[arm] == want, f"{arm}: Change log {got[arm]} differs from beta_F.json {want}")
    return got


def prereg_published():
    """The committed PREREGISTRATION.md equals the working copy and its last commit is on origin/main."""
    need(git("diff", "--quiet", "HEAD", "--", PREREG_REL).returncode == 0, "PREREGISTRATION.md has uncommitted changes")
    need(git("fetch", "origin").returncode == 0, "git fetch origin failed")
    c = git("log", "-1", "--format=%H", "--", PREREG_REL).stdout.strip()
    need(bool(c) and git("merge-base", "--is-ancestor", c, "origin/main").returncode == 0,
         f"commit {c or '?'} of PREREGISTRATION.md is not on origin/main (push it first)")
    return {"prereg_last_commit": c, "origin_main": git("rev-parse", "origin/main").stdout.strip()}


def f_betas(out, dev):
    """Prereg section 2: the betas are in the Change log and pushed before any of seeds 1000-1099 is run (any arm).
    Returns ({arm: beta or None if dropped}, info for the run log)."""
    d, p = read_beta(out, dev)
    info = {"beta_F_json_sha256": sha256(p)}
    if not dev:
        parse_beta_log(prereg_check()["change_log"], d)
        info.update(prereg_published())
    return {arm: (None if d[arm].get("dropped") else d[arm]["beta"]) for arm in F_TARGET}, info


def append_change_log(text, lines):
    """Text of PREREGISTRATION.md with the lines under '## Change log'; '(empty)' is replaced only if still there."""
    i = text.index("## Change log")
    head, tail = text[:i], text[i:]
    if tail[len("## Change log"):].strip() == "(empty)":
        return head + "## Change log\n\n" + "\n".join(lines) + "\n"
    return head + tail.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


def cmd_betalog(a):
    d, p = read_beta(a.out, a.dev)
    lines = beta_lines(d, datetime.now(KST).strftime("%Y-%m-%d"))
    print("\n".join(lines))
    if not a.append:
        return
    text = open(PREREG, encoding="utf-8").read()
    need(not re.search(r"β (F1|F2)\b", text[text.index("## Change log"):]), "the Change log already has beta lines")
    new = append_change_log(text, lines)
    need(hashlib.sha256(new[:new.index("## Change log")].encode()).hexdigest() == PREREG_BODY_SHA, "body changed")
    parse_beta_log(new[new.index("## Change log"):], d)
    dst = os.path.join(a.out, "PREREGISTRATION.dev.md") if a.dev else PREREG
    with open(f"{dst}.{os.getpid()}.tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    os.replace(f"{dst}.{os.getpid()}.tmp", dst)
    print(f"appended to {dst}" + ("" if a.dev else "; commit and push it before running seeds 1000-1099"))


def start_log(out, cmd, args, dev, est_s=None, extra=None):
    now = datetime.now(timezone.utc)
    log = {"cmd": cmd, "args": vars(args), "ns": ns_name(dev), "start": now.isoformat(), "pid": os.getpid(),
           "expected_end": (now + timedelta(seconds=est_s)).isoformat() if est_s else None,
           "expected_s": est_s, "stop_conditions": STOP_CONDITIONS, "prereg": {**prereg_check(), "commit": PREREG_COMMIT},
           "git_head": git_head(), "vf": vf_head(), "run_py_sha256": run_sha(), "versions": versions(),
           "rho_p": {str(r): rho_p(r) for r in RHOS}, "rho_p_prereg_text": {str(k): v for k, v in RHO_P_STATED.items()},
           "implementation_notes": IMPLEMENTATION_NOTES, **(extra or {})}
    tag = f"{cmd}_{args.shard.replace('/', 'of')}_{now.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    path = os.path.join(out, "logs", tag + ".json")
    write_json(path, log)
    return log, path


def end_log(log, path, **kw):
    log.update({"end": datetime.now(timezone.utc).isoformat(), **kw})
    write_json(path, log)


def unit_estimate(stage, u, timing):
    if not timing:
        return None
    if stage == 1:
        c, arm = u
        T, M, S, rho, ser = CELLS[REF if c == UNEQ else c]
        return timing["per_series_s"][f"{T}_{M}_{S}_{ser}_{arm}"] * (UNEQ_REPS if c == UNEQ else REPS1)[arm]
    scale = (timing["B"] + 1) / (B_REG + 1)   # timings are measured at B = 199
    if stage == 2:
        return timing["per_series_s"][f"{u[0]}_{u[1]}"] * S2_REPS * scale
    p = timing["parts"][u[0]]
    return (p["fit_s"] + p["boot_s"] * scale) * len(u[1])


def cmd_stage(stage, a):
    dev, out = a.dev, a.out
    check_out(out, dev)
    tp = os.path.join(out, "checks", f"timing_s{stage}.json")
    if dev:
        B = a.B or B_REG
        timing = read_json(tp) if os.path.exists(tp) else None
    else:
        need(a.n is None and a.B is None, "--n and --B are dev-only")
        timing, tp = require_checks(out, stage)
        B = timing.get("B", B_REG)
    units = units_of(stage, a)
    betas, extra = {}, {}
    if stage == 3:
        if not dev:
            require_catboost()
        if any(s >= 1000 for u in units for s in u[1]):
            betas, extra["beta_publication"] = f_betas(out, dev)
        dropped = sorted(arm for arm in betas if betas[arm] is None and any(u[0] == arm for u in units))
        if dropped:
            print(f"{dropped} dropped (no bracketing pair; beta_F.json); not run", flush=True)
            units = [u for u in units if u[0] not in dropped]
    with shard_lock(out, stage, a) as lock_path, blas_one(stage):
        extra["stage_lock"] = stage_lock(out, stage, B if stage > 1 else None, tp)
        todo = [u for u in units if not os.path.exists(part_path(out, stage, u))]
        est = [unit_estimate(stage, u, timing) for u in todo]
        extra.update({"stage": stage, "B": B if stage > 1 else None, "units": len(units), "units_todo": len(todo),
                      "shard_lock": lock_path, "blas": blas_info(),
                      "data_sha256": data_digests(dev) if stage == 3 else "Layer 1 (no data)"})
        if stage == 3:
            extra.update({"catboost_threads": catboost_threads(), "betas": betas})
        log, lp = start_log(out, f"stage{stage}", a, dev, sum(est) if est and None not in est else None, extra)
        t0, done = time.time(), 0
        for u in todo:
            path = part_path(out, stage, u)
            if os.path.exists(path):
                continue
            t = time.time()
            if stage == 1:
                rows, arrs = s1_unit(u[0], u[1], dev, a.n)
            elif stage == 2:
                rows, arrs = s2_unit(u[0], u[1], dev, B, a.n)
            else:
                rows, arrs = s3_unit(u[0], u[1][:a.n] if a.n else u[1], dev, B, betas.get(u[0]))
            save_part(path, rows, arrs)
            done += 1
            print(f"[stage{stage} {a.shard}] {u if stage < 3 else (u[0], u[1][0], u[1][-1])} {len(rows)} series "
                  f"{time.time() - t:.1f}s | {done}/{len(todo)} | elapsed {time.time() - t0:.0f}s", flush=True)
        end_log(log, lp, status="complete", units_done=done, wall_s=time.time() - t0)


# ---------------------------------------------------------------- timing (times and projections only)
def project(stage, total_s, per_s, extra):
    """Stages 1-2: 2 workers halve the wall time (one BLAS thread each). Stage 3: the B decision uses the unhalved
    projection, because CatBoost uses all cores in each worker (conservative)."""
    lim = 2 * EST_H[stage][1]
    h2, h1 = total_s / 2 / 3600, total_s / 3600
    basis = h1 if stage == 3 else h2
    out = {"stage": stage, "projected_h_2workers": h2, "projected_h_unhalved": h1,
           "decision_basis": "unhalved" if stage == 3 else "2 workers", "prereg_estimate_h": EST_H[stage],
           "limit_h": lim, "within_limit": bool(basis <= lim), "per_series_s": per_s, **extra}
    if stage == 1:
        out["decision"] = "run" if out["within_limit"] else "stop (prereg section 6)"
    return out


def cmd_time(stage, a):
    """Prints and stores times only. Uses dev-namespace seeds, so no registered series is run."""
    dev_data, out = a.dev, a.out
    check_out(out, a.dev)
    per = {}
    with blas_one(stage):
        res = _time_stage(stage, a, per, dev_data, out)
    res.update({"measured_at": datetime.now(timezone.utc).isoformat(), "blas": blas_one_info(stage),
                "versions": versions(), "vf": vf_head(), **check_stamp(a.dev)})
    write_json(os.path.join(out, "checks", f"timing_s{stage}.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_series_s", "versions", "parts")}, indent=1,
                     default=float))


def _time_stage(stage, a, per, dev_data, out):
    if stage == 1:
        l1_series("N1", 0, 0, True)
        pbo_stats(np.random.default_rng([4290]).standard_normal((1000, 8)), 16)  # warm-up
        for T, M, S, ser in product(TS, MS, SS, SERIALS):
            c = CELLS.index((T, M, S, 0.5, ser))
            for arm in ARM1:
                t = time.time()
                for j in range(2):
                    R = l1_series(arm, c, j, True)
                    pbo_stats(R, S)
                    best_full(R)
                per[f"{T}_{M}_{S}_{ser}_{arm}"] = (time.time() - t) / 2
            print(f"T={T} M={M} S={S} {ser}: " + " ".join(f"{arm} {per[f'{T}_{M}_{S}_{ser}_{arm}']:.4f}s" for arm in ARM1),
                  flush=True)
        tot = sum(per[f"{T}_{M}_{S}_{ser}_{arm}"] * n for (T, M, S, _, ser) in CELLS for arm, n in REPS1.items())
        T, M, S, _, ser = CELLS[REF]
        tot += sum(per[f"{T}_{M}_{S}_{ser}_{arm}"] * n for arm, n in UNEQ_REPS.items())
        res = project(1, tot, per, {"series": 216 * 4000 + 2500, "serial_s": tot})
    elif stage == 2:
        for c in S2_CELLS:
            for arm in S2_ARMS:
                t = time.time()
                s2_unit(c, arm, True, B_REG, n=1)
                per[f"{c}_{arm}"] = time.time() - t
            print(f"cell {c} {CELLS[c]}: " + " ".join(f"{arm} {per[f'{c}_{arm}']:.2f}s" for arm in S2_ARMS), flush=True)
        tot = sum(v * S2_REPS for v in per.values())
        res = project(2, tot, per, {"series": len(S2_CELLS) * len(S2_ARMS) * S2_REPS, "serial_s": tot,
                                    "B_measured": B_REG})
        res["B"] = B_REG if res["within_limit"] else B_FALLBACK
        res["projected_h_2workers_B99"] = res["projected_h_2workers"] * (B_FALLBACK + 1) / (B_REG + 1)
    else:
        if not dev_data:
            require_catboost()
        bfile = os.path.join(out, "beta_F.json")
        bj = read_json(bfile) if os.path.exists(bfile) else {}
        betas = {arm: (bj.get(arm) or {}).get("beta", 0.10) for arm in F_TARGET}   # 0.10: timing stand-in only
        parts = {}
        for arm in ARM2:
            if arm == "R":   # the real series is registered: time it as one A-BTC series
                per[arm], parts[arm] = per["A-BTC"], {**parts["A-BTC"], "note": "= A-BTC (R not run in timing)"}
                continue
            row = next(r for r in (l2_eval(arm, s, True, B_REG, betas.get(arm), dev_data)[0] for s in SEEDS2[arm])
                       if r["valid"])
            per[arm] = row["t_total"]
            parts[arm] = {"fit_s": row["t_fit"], "boot_s": row["t_boot"], "events": row["events"]}
            print(f"{arm}: total {row['t_total']:.1f}s (24 fits + CSCV {row['t_fit']:.1f}s, bootstrap {row['t_boot']:.1f}s, "
                  f"{row['events']} events)", flush=True)
        n = {arm: len(SEEDS2[arm]) for arm in ARM2}
        tot = sum(per[arm] * n[arm] for arm in ARM2)
        tot99 = sum((parts[arm]["fit_s"] + parts[arm]["boot_s"] * (B_FALLBACK + 1) / (B_REG + 1)) * n[arm] for arm in ARM2)
        res = project(3, tot, per, {"series": sum(n.values()), "serial_s": tot, "parts": parts,
                                    "B_measured": B_REG, "data_sha256": data_digests(dev_data),
                                    "betas_used_for_timing": betas, "catboost_threads": catboost_threads()})
        res["B"] = B_REG if res["within_limit"] else B_FALLBACK
        res["projected_h_unhalved_B99"], res["projected_h_2workers_B99"] = tot99 / 3600, tot99 / 2 / 3600
    return res


# ---------------------------------------------------------------- calibration of beta for F1, F2
def interp_beta(rows, target):
    for a, b in zip(rows, rows[1:]):
        if (a["mean_auc"] - target) * (b["mean_auc"] - target) <= 0:
            x = a["beta"] if a["mean_auc"] == b["mean_auc"] else \
                a["beta"] + (target - a["mean_auc"]) * (b["beta"] - a["beta"]) / (b["mean_auc"] - a["mean_auc"])
            return {"target": target, "pair": [a["beta"], b["beta"]], "beta_exact": x, "beta": round(x, 3)}
    return {"dropped": True, "target": target,
            "reason": f"no adjacent grid pair brackets mean holdout AUC {target:.2f} up to beta {BETA_GRID[-1]:.3f}"}


def cmd_calib(a):
    dev, out = a.dev, a.out
    check_out(out, dev)
    if not dev:
        require_catboost()
    log, lp = start_log(out, "calib", a, dev, None, {"data_sha256": data_digests(dev)})
    dfa, r = returns("BTC", dev)
    base = e15.demean(r)
    rows = []
    for beta in BETA_GRID:
        aucs = []
        for s in CALIB_SEEDS:
            df, ret = make_series_w(rs_seed(dev, s), base, len(base), dfa.index, beta, F_WINDOW)
            p = e14.prepare(df, ret)
            if p is None:
                continue
            X, y, fwd, split = p
            aucs.append(e15.gate(X, y, fwd, split, split)["auc"])
        rows.append({"beta": beta, "n": len(aucs), "mean_auc": float(np.mean(aucs)), "aucs": aucs})
        print({k: v for k, v in rows[-1].items() if k != "aucs"}, flush=True)
    res = {**check_stamp(dev), "window": F_WINDOW, "seeds": "2000-2019", "grid": rows,
           "rule": "first bracketing adjacent pair from beta = 0, linear interpolation, beta rounded to 3 decimals"}
    for arm, tgt in F_TARGET.items():
        res[arm] = interp_beta(rows, tgt)
    write_json(os.path.join(out, "beta_F.json"), res)
    print(json.dumps({k: res[k] for k in F_TARGET}, indent=1))
    print("next: run.py betalog --append, then commit and push PREREGISTRATION.md")
    end_log(log, lp, status="complete", betas={k: res[k] for k in F_TARGET})


# ---------------------------------------------------------------- replication (real data only)
def block_stats(R, S):
    blocks = np.array_split(np.arange(R.shape[0]), S)
    return (np.array([len(b) for b in blocks]), np.stack([R[b].sum(0) for b in blocks]),
            np.stack([(R[b] ** 2).sum(0) for b in blocks]))


def cmd_repl_dev(a):
    """Dev proxy of the replication (synthetic data, dev seeds): the M = 8 columns of the 24-grid equal E15's own
    cv6_returns(GRID_P, make_p) bitwise. Writes checks/repl_dev.json; it is not the prereg replication."""
    check_out(a.out, True)
    res, ok = [], True
    for s in (0, 1):
        X, y, fwd, split = e15.build("A", "BTC", rs_seed(True, s), cache(True))
        R24 = e15.cv6_returns(X, fwd, y, GRID24, make_c)
        R8 = e15.cv6_returns(X, fwd, y, e15.GRID_P, e15.make_p)
        d = {"dev_seed": s, "events": len(X), "M8_cols_bitwise_equal": bool(np.array_equal(R24[:, SUB[8]], R8)),
             "block_stats_equal": all(np.array_equal(u, v) for u, v in zip(block_stats(R24[:, SUB[8]], 16),
                                                                           block_stats(R8, 16)))}
        ok &= d["M8_cols_bitwise_equal"] and d["block_stats_equal"]
        res.append(d)
        print(json.dumps(d), flush=True)
    write_json(os.path.join(a.out, "checks", "repl_dev.json"), {"pass": bool(ok), "seeds": res, **check_stamp(True)})
    print("REPL-DEV PASS" if ok else "REPL-DEV FAIL")


def cmd_repl(a):
    out = a.out
    check_out(out, False)
    require_catboost()
    log, lp = start_log(out, "repl", a, False, None,
                        {"data_sha256": data_digests(False), "catboost_threads": catboost_threads()})
    z = np.load(os.path.join(ROOT, "results", "inbox7", "stage1_arrays.npz"))
    res, ok = [], True
    for s, target in zip(range(5), REPL_COUNTS):
        X, y, fwd, split = e15.build("A", "BTC", s, cache(False))
        R = e15.cv6_returns(X, fwd, y, GRID24, make_c)[:, SUB[8]]
        n, s1, s2 = block_stats(R, 16)
        k = f"A_BTC_{s}_S3"
        d = {"seed": s, "events": len(X), "n_equal": bool(np.array_equal(n, z[k + "_n"])),
             "sum_maxdiff": float(np.abs(s1 - z[k + "_sum"]).max()),
             "sumsq_maxdiff": float(np.abs(s2 - z[k + "_sumsq"]).max()),
             "pbo_count": int(round(e15.pbo_fast(*e15.cscv_matrices(R, 16)) * 12870)), "pbo_count_ref": target}
        d["pass"] = bool(d["n_equal"] and d["sum_maxdiff"] <= 1e-12 and d["sumsq_maxdiff"] <= 1e-12
                         and d["pbo_count"] == target)
        ok &= d["pass"]
        res.append(d)
        print(json.dumps(d), flush=True)
    write_json(os.path.join(out, "checks", "repl.json"), {"pass": bool(ok), "seeds": res, "log": lp,
                                                         "vf": vf_head(), **check_stamp(False)})
    end_log(log, lp, status="complete", passed=bool(ok))
    print("REPL PASS" if ok else "REPL FAIL")


# ---------------------------------------------------------------- unit tests (prereg section 6 + implementation)
def _ties_pbo():
    """experiments/inbox7/ties.py pbo(), extracted without running the script's module-level code."""
    import ast
    src = open(os.path.join(HERE, "..", "inbox7", "ties.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "pbo")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "RULES" for t in n.targets))]
    ns = {"np": np, "rankdata": rankdata}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "ties.py", "exec"), ns)
    return ns["pbo"]


def cmd_unit(a):
    out = a.out
    check_out(out, a.dev)
    res, ok = {}, {}
    t_all = time.time()
    # 1. CSCV: one column dominates every block -> 0; identical columns -> 1 (E15 functions, unchanged)
    rng = np.random.default_rng([4242, 1])
    dom, ide = [], []
    for S in SS:
        for M in (4, 8, 32):
            R = rng.standard_normal((1000, M))
            R[:, M // 2] = np.abs(R[:, M // 2]) + 5.0
            dom.append(e15.pbo_fast(*e15.cscv_matrices(R, S)))
            ide.append(e15.pbo_fast(*e15.cscv_matrices(np.repeat(rng.standard_normal((1000, 1)), M, axis=1), S)))
    res["cscv_dominant"], res["cscv_identical"] = dom, ide
    ok["cscv_dominant_0"] = all(v == 0.0 for v in dom)
    ok["cscv_identical_1"] = all(v == 1.0 for v in ide)
    # 2. generator: pairwise Pearson correlation of e at T = 1e6 within 0.01 of the target
    rp = {r: rho_p(r) for r in RHOS}
    res["rho_p"] = rp
    res["rho_p_minus_prereg_text"] = {r: rp[r] - RHO_P_STATED[r] for r in RHOS}
    ok["rho_p_matches_prereg_text_0.001"] = all(abs(v) < 1e-3 for v in res["rho_p_minus_prereg_text"].values())
    corr = {}
    for si, ser in enumerate(SERIALS):
        for ri, rho in enumerate(RHOS):
            e = shocks(np.random.default_rng([4243, si, ri]), 10 ** 6, 8, rho, ser)
            c = np.corrcoef(e, rowvar=False)[np.triu_indices(8, 1)]
            tgt = rp[rho] if ser == "GARCH" else rho
            corr[f"{ser}_{rho}"] = {"target": tgt, "max_abs_dev": float(np.abs(c - tgt).max()),
                                    "var_e_range": [float(e.var(0).min()), float(e.var(0).max())]}
            del e
    res["generator_corr_T1e6"] = corr
    ok["generator_corr_within_0.01"] = all(v["max_abs_dev"] <= 0.01 for v in corr.values())
    # 3. annual SR of each column within 0.1 of target (M = 8, arm P2; T = 1e6 IID / GARCH, 1e7 MA10)
    srs = {}
    sig = sigmas(8)
    tgt = ann_sr("P2", 8)
    for si, ser in enumerate(SERIALS):
        for ri, rho in enumerate(RHOS):
            T = 10 ** 7 if ser == "MA10" else 10 ** 6
            r = shocks(np.random.default_rng([4244, si, ri]), T, 8, rho, ser)
            r *= sig
            r += means("P2", sig)
            sr = r.mean(0) / r.std(0) * math.sqrt(ANN)
            srs[f"{ser}_{rho}_T{T:.0e}"] = {"max_abs_dev": float(np.abs(sr - tgt).max()), "sr_col1": float(sr[0])}
            del r
    res["generator_annual_sr"] = srs
    ok["generator_sr_within_0.1"] = all(v["max_abs_dev"] <= 0.1 for v in srs.values())
    # 4. demeaning: column means of the matrix passed to the bootstrap (demeaned(), as calibrated() calls it) are 0
    dmx = []
    for arm, c in (("N2", 0), ("P3", REF), ("N3", REF), ("P1", CELLS.index((2500, 32, 8, 0.9, "MA10"))), ("P2", UNEQ)):
        dmx.append(float(np.abs(demeaned(l1_series(arm, c, 0, True)).mean(0)).max()))
    Rl2 = np.where(np.random.default_rng([4245]).random((900, 24)) < 0.5, 1.0, -1.0) * \
        np.random.default_rng([4246]).standard_t(3, (900, 1)) * 0.04
    dmx.append(float(np.abs(demeaned(Rl2).mean(0)).max()))
    res["demeaned_abs_mean_max"] = dmx
    ok["demean_1e-12"] = max(dmx) <= 1e-12
    # 5. no re-centring: 10,000 resamples of one N1 matrix (reference-cell shape) through boot(), as calibrated()
    T, M, S, rho, ser = CELLS[REF]
    R = means("N1", sigmas(M)) + sigmas(M) * shocks(np.random.default_rng([4247]), T, M, rho, ser)
    D = demeaned(R)
    mb = np.array([boot(D, [4248, b], (L0,))[1][L0].mean(0) for b in range(10000)])
    sd, avg = mb.std(0, ddof=1), mb.mean(0)
    res["no_recentring"] = {"sd_min": float(sd.min()), "max_abs_avg_over_se": float(np.max(np.abs(avg) / (sd / 100)))}
    ok["no_recentring"] = bool(sd.min() > 0 and res["no_recentring"]["max_abs_avg_over_se"] <= 3)
    # 6. p in [1/(B+1), 1] (asserted in calibrated(); exercised on four matrices incl. extremes)
    ps = []
    Rd = rng.standard_normal((400, 8))
    Rd[:, 2] = np.abs(Rd[:, 2]) + 5.0
    for R_, S_ in ((l1_series("N1", 1, 0, True), 8), (l1_series("P3", 1, 0, True), 8), (Rd, 8),
                   (np.repeat(rng.standard_normal((400, 1)), 8, axis=1), 8)):
        r_, _ = calibrated(R_, S_, [[4249, b] for b in range(B_REG)], (L0,) + L_SENS)
        ps += [r_[f"p_L{L}"] for L in (L0,) + L_SENS] + [r_["p_rc"]]
    res["p_values"] = ps
    ok["p_in_range"] = all(1 / (B_REG + 1) <= p <= 1 for p in ps)
    # implementation tests -------------------------------------------------------------------
    # stationary bootstrap: vectorised == E15 loop; and == make_sb_series on its RandomState draws
    eq = True
    for n in (5, 250, 937, 1000, 2500):
        for L in (L0,) + L_SENS:
            for b in range(5):
                st = sb_draw(np.random.default_rng([4250, n, L, b]), n)
                eq &= bool(np.array_equal(sb_index(*st, n, L), sb_loop(*st, n, L)))
    ok["sb_vectorised_eq_loop"] = eq
    idx_ref = pd.date_range("2017-01-01", periods=1000, freq="D")
    eq = True
    for seed in (0, 7, 123):
        _, _, idx15 = e15.make_sb_series(seed, rng.standard_normal(1000) * 0.03, 1000, idx_ref)
        rs = np.random.RandomState(seed)
        st = (rs.randint(1000), rs.random_sample(999), rs.randint(0, 1000, size=999))
        eq &= bool(np.array_equal(sb_index(*st, 1000, L0), idx15))
    ok["sb_eq_e15_make_sb_series"] = eq
    # RC from the original matrix == from the demeaned matrix (same indices)
    R = l1_series("P2", 1, 0, True)
    st = sb_draw(np.random.default_rng([4251]), R.shape[0])
    idx = sb_index(*st, R.shape[0], L0)
    res["rc_orig_vs_demeaned_maxdiff"] = float(np.abs((R[idx].mean(0) - R.mean(0)) - (R - R.mean(0))[idx].mean(0)).max())
    ok["rc_orig_vs_demeaned"] = res["rc_orig_vs_demeaned_maxdiff"] < 1e-15
    # make_series_w(window=5) == e14.make_series bitwise; window 40 recursion
    base = rng.standard_normal(3132) * 0.035
    base -= base.mean()
    ix = pd.date_range("2017-09-01", periods=3133, freq="D")
    eq = True
    for seed in (0, 1, 1000, 2000, 99000000):
        for beta in (None, 0.0, 0.117, 0.163):
            d1, r1 = e14.make_series(seed, base, len(base), ix, beta)
            d2, r2 = make_series_w(seed, base, len(base), ix, beta, 5)
            eq &= bool(np.array_equal(r1, r2) and np.array_equal(d1.values, d2.values) and d1.index.equals(d2.index))
    ok["make_series_w5_eq_e14"] = eq
    d0, r0 = make_series_w(4252, base, len(base), ix, None, 40)
    d4, r4 = make_series_w(4252, base, len(base), ix, 0.10, 40)
    mu, sg = r0.mean(), r0.std()
    z = np.array([0.0 if t < 39 else (r4[t - 39:t + 1].sum() - 40 * mu) / (sg * np.sqrt(40)) for t in range(len(r0) - 1)])
    res["z40_recursion_maxdiff"] = float(np.abs((r4[1:] - r0[1:]) - 0.10 * sg * z).max())
    ok["z40_recursion"] = bool(res["z40_recursion_maxdiff"] < 1e-15 and np.array_equal(r4[:40], r0[:40])
                               and np.array_equal(make_series_w(4252, base, len(base), ix, 0.0, 40)[1], r0))
    # grids: M = 8 subset == E15 GRID_P order; sub-grid definitions; constructor == make_p + l2_leaf_reg
    ok["grid_M8_eq_GRID_P"] = [GRID24[i][:3] for i in SUB[8]] == list(e15.GRID_P) and all(GRID24[i][3] == 3 for i in SUB[8])
    ok["grid_subsets"] = (len(GRID24) == 24 and SUB[16] == list(range(16))
                          and [GRID24[i] for i in SUB[4]] == [(4, .03, None, 3), (4, .1, None, 3), (6, .03, None, 3), (6, .1, None, 3)]
                          and sorted(set(g[0] for g in GRID24)) == [4, 6, 8])
    same = True
    for g in GRID24:
        pc, pp = make_c(g).get_params(), e14.make_p(g[:3]).get_params()
        same &= pc.pop("l2_leaf_reg") == g[3] and pc == pp
    ok["make_c_eq_make_p_plus_l2"] = same
    # tie rules == ties.py per-split rule; first == pbo_fast (random, integer-tied, CSCV with equal columns)
    tp = _ties_pbo()
    eq = True
    for i in range(60):
        r_ = np.random.default_rng([4253, i])
        M_, K_ = int(r_.integers(2, 12)), int(r_.integers(1, 80))
        if i % 3 == 0:
            im, om = r_.standard_normal((M_, K_)), r_.standard_normal((M_, K_))
        elif i % 3 == 1:
            im, om = r_.integers(0, 3, (M_, K_)).astype(float), r_.integers(0, 3, (M_, K_)).astype(float)
        else:
            Rr = np.where(r_.random((300, 1)) < 0.5, 1.0, -1.0) * r_.standard_normal((300, 1))
            Rr = np.repeat(Rr, M_, axis=1)
            Rr[:, ::2] = np.where(r_.random((300, 1)) < 0.5, 1.0, -1.0) * np.abs(Rr[:, ::2])
            im, om = e15.cscv_matrices(Rr, 8)
            K_ = im.shape[1]
        t1 = tie_rules(im, om)
        rr, tot, tie, _ = tp(im, om)
        eq &= all(t1[k] / K_ == rr[k] for k in ("first", "last", "mean_rank")) and t1["tie"] == tie and tot == K_
        eq &= t1["first"] / K_ == e15.pbo_fast(im, om)
    ok["tie_rules_eq_ties_py"] = eq
    # BLAS thread count does not change any Layer 1 statistic (stages 1-2 run with one BLAS thread)
    eq = True
    for arm, c in (("N1", REF), ("P2", REF), ("N2", CELLS.index((2500, 32, 16, 0.9, "MA10")))):
        R = l1_series(arm, c, 1, True)
        S_ = CELLS[c][2]
        st = [[4254, b] for b in range(20)]
        d_def = (pbo_stats(R, S_), calibrated(R, S_, st, (L0,) + L_SENS))
        with threadpool_limits(limits=1, user_api="blas"):
            d_one = (pbo_stats(R, S_), calibrated(R, S_, st, (L0,) + L_SENS))
        eq &= d_def[0] == d_one[0] and d_def[1][0] == d_one[1][0] and all(
            np.array_equal(d_def[1][1][L], d_one[1][1][L]) for L in d_def[1][1])
    ok["blas_threads_invariant"] = bool(eq)
    res["blas"] = blas_info()
    # Change-log beta lines: generated lines parse back to beta_F.json; malformed, duplicate, differing lines refuse
    toy = {"F1": {"beta": 0.066, "beta_exact": 0.0657312, "target": 0.56}, "F2": {"dropped": True, "target": 0.60}}
    pre = open(PREREG, encoding="utf-8").read()
    lines = beta_lines(toy, "2026-10-24")
    new = append_change_log(pre, lines)
    body_kept = new[:new.index("## Change log")] == pre[:pre.index("## Change log")]
    refused = []
    for bad in (lines + [lines[0]], [lines[0]], [lines[0].replace("0.066", "0.067"), lines[1]],
                [lines[0].replace("β F1 =", "β F1="), lines[1]], [lines[0], lines[1].replace("dropped", "drop")]):
        try:
            parse_beta_log("\n".join(bad), toy)
            refused.append(False)
        except SystemExit:
            refused.append(True)
    res["betalog_lines"] = lines
    ok["betalog_roundtrip"] = bool(body_kept and parse_beta_log(new[new.index("## Change log"):], toy)["F1"]["beta"]
                                   == "0.066" and "(empty)" not in new[new.index("## Change log"):] and all(refused)
                                   and append_change_log(new, ["- x"]).endswith(lines[1] + "\n- x\n"))
    # cells and pairing
    ok["cells"] = (len(CELLS) == 216 and REF == 103 and CELLS[REF] == (1000, 8, 16, 0.5, "GARCH")
                   and len(S2_CELLS) == 19 and CELLS[0] == (250, 4, 8, 0.0, "IID") and CELLS[1][4] == "GARCH")
    p2, n3 = l1_series("P2", REF, 3, True), l1_series("N3", REF, 3, True)
    edge = EDGE["P2"] / math.sqrt(ANN) * SIGMA
    res["n3_pair_maxdev"] = float(np.abs((p2 - n3) - edge).max())
    ok["n3_pairs_p2"] = res["n3_pair_maxdev"] < 1e-17
    su = sigmas(8, True)
    ok["uneq_sigma"] = bool(su[0] == 0.005 and abs(su[-1] - 0.02) < 1e-17 and np.all(np.diff(su) > 0))
    # reference counts of the replication: stored E15 block arrays give the prereg's PBO counts
    spec = importlib.util.spec_from_file_location("rc15", os.path.join(HERE, "..", "inbox7", "recompute.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    z = np.load(os.path.join(ROOT, "results", "inbox7", "stage1_arrays.npz"))
    cnt = [int(round(rc.cscv_pbo(z[f"A_BTC_{s}_S3_n"], z[f"A_BTC_{s}_S3_sum"], z[f"A_BTC_{s}_S3_sumsq"])[0] * 12870))
           for s in range(5)]
    res["repl_reference_counts"] = cnt
    ok["repl_reference_counts"] = tuple(cnt) == REPL_COUNTS
    res["wall_s"] = time.time() - t_all
    passed = all(ok.values())
    print(json.dumps({"results": res, "checks": ok}, indent=1, default=float))
    write_json(os.path.join(out, "checks", "unit.json"), {"pass": passed, "checks": ok, "results": res,
                                                         "versions": versions(), "prereg": prereg_check(),
                                                         "vf": vf_head(), **check_stamp(a.dev)})
    print("UNIT PASS" if passed else "UNIT FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unit", "repl", "time", "calib", "betalog", "stage1", "stage2", "stage3"])
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    if a.out is None:
        need(not a.dev, "--dev requires --out outside results/")
        a.out = OUT_DEFAULT
    need(a.dev or (a.n is None and a.B is None), "--n and --B are dev-only")
    prereg_check()
    vf_head()
    if a.cmd == "unit":
        cmd_unit(a)
    elif a.cmd == "repl":
        cmd_repl_dev(a) if a.dev else cmd_repl(a)
    elif a.cmd == "time":
        cmd_time(a.stage, a)
    elif a.cmd == "calib":
        cmd_calib(a)
    elif a.cmd == "betalog":
        check_out(a.out, a.dev)
        cmd_betalog(a)
    else:
        cmd_stage(int(a.cmd[-1]), a)

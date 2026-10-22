"""Robustness study of the E14 findings (CHANGELOG E15).

Design: valid-framework experiments/inbox7/PREREGISTRATION.md (public, bae70d3,
sha256 7ee7d405274d60f7cbda05989920b0f999baeb1730bd617726f98167a441fad2). Nothing here may
change the design; every implementation detail the pre-registration leaves open is listed in
DEVIATIONS below and in CHANGELOG E15.

The E14 pipeline is imported unchanged from experiments/true_pbo/run.py (series generator, prepare,
cpcv_paths, grid_matrices, pbo_eq1, base model, cost rule), which in turn imports the
valid-framework worktree at 018f4ea (VF env var, default /tmp/vf-pbo).

Arms (prereg section 1): A (demeaned i.i.d.), B0 / B1 / B2 (demeaned, beta 0 / 0.117 / 0.163),
B0p (= B0', E14 base not demeaned, beta 0), C (demeaned stationary bootstrap, mean block 20).
Statistics (prereg section 2): G, G', S1-S4, T1-T4.

usage (python3 -I, from the repository root):
  run.py unit
  run.py repl                                   # S1 + T1 on the E14 base, BTC seeds 0-4
  run.py time    --stage 1|2
  run.py series  --stage 1|2 --arm A|B0|B1|B2|B0p|C [--asset BTC] [--seeds a-b] --out PREFIX
"""
import argparse
import importlib.util
import json
import os
import time
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("e14", os.path.join(HERE, "..", "true_pbo", "run.py"))
e14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e14)  # sets up the VF worktree imports

np, pd = e14.np, e14.pd
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PURGE = e14.config.CPCV_PURGE          # 20 events (CPCV, CV6, validation slice)
HOLD_PURGE = 20                        # T2-T4 and purged G: training events dropped before the holdout
N_PERM = 100
BLOCK = 20                             # T4 block length (events)
SB_P = 1 / 20                          # C: stationary bootstrap, geometric block length, mean 20 days
CSCV_S = 16
COST_EVERY = 18 / 10000                # G': 18 bp at every event
EPS = e14.EPS
assert e14.COST_RT == COST_EVERY       # E14 cost rule uses the same 18 bp round trip

GRID_P, make_p = e14.GRID_P, e14.make_p
GRID_L = [(nl, lr, cw) for nl in (15, 63) for lr in (0.03, 0.1) for cw in (None, "balanced")]  # S4, M = 8

# arm -> (demean, generator, beta, prereg seeds)
ARMS = {
    "A": (True, "iid", None, "0-199"),
    "B0": (True, "iid", None, "1000-1099"),
    "B1": (True, "iid", 0.117, "1000-1099"),
    "B2": (True, "iid", 0.163, "1000-1099"),
    "B0p": (False, "iid", None, "1000-1099"),
    "C": (True, "sb", None, "0-199"),
}
STAGE_ARMS = {1: ["A", "B0", "B1", "B2", "B0p", "C"], 2: ["A", "B0", "B1", "B2"]}
STAGE_SERIES = {1: 800, 2: 900}
STAGE_EST_H = {1: 3.6, 2: 4.0}

DEVIATIONS = [
    "beta = 0 (B0, B0p) is generated with beta=None; the E14 drift loop with beta=0 adds exactly 0.0 "
    "(unit test: identical arrays)",
    "C: RandomState(seed); start index randint(n); then n_days-1 uniforms and n_days-1 candidate "
    "starts are drawn as two vectors; a new block starts when u < 1/20, else the index advances by 1 "
    "modulo n (wrap-around); the OHLC noise then follows from the same generator as in E14",
    "S2: training indices of the split in time order; validation = positions int(0.8 n_tr): (last "
    ">= 20%, holdout convention of E14); fit = positions [: int(0.8 n_tr) - 20]",
    "S3: 6 groups as in cpcv_paths; group g predicted by a model trained on the other five minus 20 "
    "events on each side of g; 16 row blocks = np.array_split (sizes differ by <= 1); per-half Sharpe "
    "from block sums (mean / (sqrt(E[x^2]-mean^2) + 1e-10)), equal to sr_pe up to rounding; IS-best "
    "ties -> first configuration (np.argmax, as E14)",
    "S4: LightGBM defaults otherwise (max_depth -1, min_child_samples 20), verbose -1",
    "PBO = NaN (no valid path) counts as not passing a PBO threshold",
    "G purged: base model fitted on events [: split - 20], holdout unchanged",
    "T3: offsets = RandomState(seed*1000+100).choice(range(floor(0.1n), ceil(0.9n)+1), 100, "
    "replace=False), labels rolled with np.roll",
    "T4: blocks of 20 from the first event; block order = RandomState(seed*1000+200+i).permutation",
    "T1/T2 permutations: RandomState(seed*1000+i).permutation(y), i = 0..99 (E14 convention); T2 uses "
    "the same permuted vectors as T1",
    "G': (mean - 0.0018) / (sd + 1e-10) of the per-event holdout returns, on both holdouts",
]


# ---------------------------------------------------------------- series
def demean(r):
    return r - r.mean()


def make_sb_series(seed, ret, n_days, ref_index, p=SB_P):
    """Stationary bootstrap (Politis and Romano 1994) of ret; OHLC as in E14 make_series."""
    rng = np.random.RandomState(seed)
    n = len(ret)
    idx = np.empty(n_days, dtype=int)
    idx[0] = rng.randint(n)
    u = rng.random_sample(n_days - 1)
    jump = rng.randint(0, n, size=n_days - 1)
    for t in range(1, n_days):
        idx[t] = jump[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
    r = np.asarray(ret)[idx]
    price = 10000 * np.exp(np.cumsum(r))
    df = pd.DataFrame({
        "open": price,
        "high": price * (1 + np.abs(rng.normal(0, 0.01, n_days))),
        "low": price * (1 - np.abs(rng.normal(0, 0.01, n_days))),
        "close": price,
        "volume": rng.lognormal(20, 1, n_days),
    }, index=ref_index[:n_days])
    return df, r, idx


def build(arm, asset, seed, cache={}):
    if asset not in cache:
        cache[asset] = e14.asset_returns(asset)
    dfa, r = cache[asset]
    dm, gen, beta, _ = ARMS[arm]
    base = demean(r) if dm else r
    if gen == "sb":
        df, ret, _ = make_sb_series(seed, base, len(base), dfa.index)
    else:
        df, ret = e14.make_series(seed, base, len(base), dfa.index, beta)
    return e14.prepare(df, ret)


# ---------------------------------------------------------------- models
def make_l(cfg):
    from lightgbm import LGBMClassifier
    nl, lr, cw = cfg
    return LGBMClassifier(n_estimators=100, num_leaves=nl, learning_rate=lr, class_weight=cw,
                          random_state=42, n_jobs=1, verbose=-1)


def make_base():
    return e14.get_model_fn("catboost", True, iterations=100)[0]()


def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except Exception:
        return 0.5


# ---------------------------------------------------------------- statistics
def gate(X, y, fwd, split, tr_end):
    """E14 base_gate without permutations, fitted on [:tr_end]; plus G'."""
    m = make_base()
    m.fit(X[:tr_end], y[:tr_end])
    auc = safe_auc(y[split:], m.predict_proba(X[split:])[:, 1])
    preds = np.asarray(m.predict(X[split:])).ravel()
    strat = np.where(preds == 1, 1.0, -1.0) * fwd[split:]
    trades = np.abs(np.diff(np.concatenate([[0], preds]))).sum()
    cost = trades * e14.COST_RT / len(preds)
    net = (strat.mean() - cost) / (strat.std() + EPS)
    net_every = (strat.mean() - COST_EVERY) / (strat.std() + EPS)
    return {"auc": auc, "net_sr_pe": net, "net_every_sr_pe": net_every,
            "gross_sr_pe": strat.mean() / (strat.std() + EPS), "long_pct": float((preds == 1).mean())}


def grid_matrices_val(X, fwd, y, paths, grid, maker, min_te=5, min_tr=10):
    """S2: IS = Sharpe on the last 20% of the training events, fit on the rest minus 20."""
    M, J = len(grid), len(paths)
    is_m, oos_m = np.full((M, J), np.nan), np.full((M, J), np.nan)
    for j, (tr_i, te_i) in enumerate(paths):
        v0 = int(len(tr_i) * 0.8)
        val, fit = tr_i[v0:], tr_i[:max(0, v0 - PURGE)]
        if len(te_i) < min_te or len(fit) < min_tr or len(val) < 2:
            continue
        for ci, cfg in enumerate(grid):
            m = maker(cfg)
            m.fit(X[fit], y[fit])
            p_v = np.asarray(m.predict(X[val])).ravel()
            p_te = np.asarray(m.predict(X[te_i])).ravel()
            is_m[ci, j] = e14.sr_pe(np.where(p_v == 1, 1.0, -1.0) * fwd[val])
            oos_m[ci, j] = e14.sr_pe(np.where(p_te == 1, 1.0, -1.0) * fwd[te_i])
    return is_m, oos_m


def group_ids(n):
    gs = n // 6
    gids = np.zeros(n, dtype=int)
    for g in range(6):
        gids[g * gs:(g + 1) * gs if g < 5 else n] = g
    return gids


def cv6_returns(X, fwd, y, grid, maker, purge=PURGE):
    """S3 input: T x M per-event OOS returns from purged 6-fold CV."""
    n = len(X)
    gids = group_ids(n)
    R = np.full((n, len(grid)), np.nan)
    for g in range(6):
        te = np.where(gids == g)[0]
        s, e = te[0], te[-1]
        pm = np.zeros(n, dtype=bool)
        pm[max(0, s - purge):s] = True
        pm[e + 1:min(n, e + purge + 1)] = True
        tr = np.where((gids != g) & ~pm)[0]
        for ci, cfg in enumerate(grid):
            m = maker(cfg)
            m.fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[te])).ravel()
            R[te, ci] = np.where(p == 1, 1.0, -1.0) * fwd[te]
    return R


_CSCV_IND = {}


def cscv_halves(S):
    if S not in _CSCV_IND:
        cs = list(combinations(range(S), S // 2))
        ind = np.zeros((len(cs), S))
        for i, c in enumerate(cs):
            ind[i, list(c)] = 1.0
        _CSCV_IND[S] = ind
    return _CSCV_IND[S]


def cscv_matrices(R, S=CSCV_S):
    """Canonical CSCV (Bailey et al. 2017): all C(S, S/2) halves of S contiguous row blocks.
    Returns IS and OOS Sharpe matrices of shape (M, C(S, S/2))."""
    blocks = np.array_split(np.arange(R.shape[0]), S)
    nb = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([R[b].sum(0) for b in blocks])          # S x M
    s2 = np.stack([(R[b] ** 2).sum(0) for b in blocks])
    ind = cscv_halves(S)

    def sr(I):
        n = (I @ nb)[:, None]
        mu = (I @ s1) / n
        var = np.maximum((I @ s2) / n - mu ** 2, 0.0)
        return mu / (np.sqrt(var) + EPS)

    return sr(ind).T, sr(1.0 - ind).T


def pbo_fast(is_m, oos_m):
    """pbo_eq1 for matrices without NaN (CSCV): same rule, vectorised."""
    M = is_m.shape[0]
    best = is_m.argmax(0)
    ranks = rankdata(oos_m, axis=0)
    w = ranks[best, np.arange(is_m.shape[1])] / (M + 1)
    return float(np.mean(np.log(w / (1 - w)) <= 0))


def permutations_full(y, seed):
    return [np.random.RandomState(seed * 1000 + i).permutation(y) for i in range(N_PERM)]


def permutations_shift(y, seed):
    n = len(y)
    offs = np.random.RandomState(seed * 1000 + 100).choice(
        np.arange(int(np.floor(0.1 * n)), int(np.ceil(0.9 * n)) + 1), N_PERM, replace=False)
    return [np.roll(y, int(k)) for k in offs], offs


def block_shuffle(y, rng, block=BLOCK):
    blocks = [np.arange(i, min(i + block, len(y))) for i in range(0, len(y), block)]
    order = rng.permutation(len(blocks))
    return y[np.concatenate([blocks[o] for o in order])]


def permutations_block(y, seed):
    return [block_shuffle(y, np.random.RandomState(seed * 1000 + 200 + i)) for i in range(N_PERM)]


def perm_aucs(X, split, tr_end, perms):
    out = []
    for ys in perms:
        m = make_base()
        m.fit(X[:tr_end], ys[:tr_end])
        out.append(safe_auc(ys[split:], m.predict_proba(X[split:])[:, 1]))
    return np.array(out)


def pbo_row(row, arrs, tag, is_m, oos_m):
    p, _, n = e14.pbo_eq1(is_m, oos_m)
    row[f"pbo_{tag}"], row[f"paths_{tag}"] = p, n
    arrs[f"{tag}_is"], arrs[f"{tag}_oos"] = is_m, oos_m


def eval_stage1(X, y, fwd, split, seed):
    row, arrs = {"events": len(X), "split": split}, {}
    t = time.time()
    for h, tr_end in (("u", split), ("p", split - HOLD_PURGE)):
        g = gate(X, y, fwd, split, tr_end)
        row.update({f"{k}_{h}": v for k, v in g.items()})
        row[f"G_{h}"] = bool(g["auc"] > 0.55 and g["net_sr_pe"] > 0)
    row["t_G"] = time.time() - t
    paths = e14.cpcv_paths(len(X), PURGE)
    t = time.time()
    pbo_row(row, arrs, "S1", *e14.grid_matrices(X, fwd, y, paths, GRID_P, make_p))
    row["t_S1"] = time.time() - t
    t = time.time()
    pbo_row(row, arrs, "S2", *grid_matrices_val(X, fwd, y, paths, GRID_P, make_p))
    row["t_S2"] = time.time() - t
    t = time.time()
    R = cv6_returns(X, fwd, y, GRID_P, make_p)
    arrs["S3_R"] = R
    i3, o3 = cscv_matrices(R)
    row["pbo_S3"], row["paths_S3"] = pbo_fast(i3, o3), i3.shape[1]
    row["t_S3"] = time.time() - t
    t = time.time()
    pbo_row(row, arrs, "S4", *e14.grid_matrices(X, fwd, y, paths, GRID_L, make_l))
    row["t_S4"] = time.time() - t
    return row, arrs


def eval_stage2(X, y, fwd, split, seed):
    row, arrs = {"events": len(X), "split": split}, {}
    t = time.time()
    for h, tr_end in (("u", split), ("p", split - HOLD_PURGE)):
        g = gate(X, y, fwd, split, tr_end)
        row.update({f"{k}_{h}": v for k, v in g.items()})
        row[f"G_{h}"] = bool(g["auc"] > 0.55 and g["net_sr_pe"] > 0)
        row[f"Gp_{h}"] = bool(g["net_every_sr_pe"] > 0)
    row["t_G"] = time.time() - t
    full = permutations_full(y, seed)
    shift, offs = permutations_shift(y, seed)
    arrs["T3_offsets"] = offs
    for tag, perms, h in (("T1", full, "u"), ("T2", full, "p"), ("T3", shift, "p"),
                          ("T4", permutations_block(y, seed), "p")):
        t = time.time()
        a = perm_aucs(X, split, split if h == "u" else split - HOLD_PURGE, perms)
        arrs[f"{tag}_aucs"] = a
        row[f"{tag}_p95"] = float(np.percentile(a, 95))
        row[f"{tag}_pass"] = bool(row[f"auc_{h}"] > row[f"{tag}_p95"])
        row[f"t_{tag}"] = time.time() - t
    return row, arrs


# ---------------------------------------------------------------- commands
def seeds_of(s):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


def cmd_unit():
    res = {}
    rng = np.random.RandomState(0)
    # CSCV: one column dominates on every block -> PBO 0
    R = rng.normal(0, 1, size=(800, 8))
    R[:, 3] = np.abs(R[:, 3]) + 5.0
    res["cscv_dominant"] = pbo_fast(*cscv_matrices(R))
    # identical columns -> PBO 1 (averaged ties: rank (M+1)/2, logit 0 counts)
    c = rng.normal(0, 1, size=(800, 1))
    res["cscv_identical"] = pbo_fast(*cscv_matrices(np.repeat(c, 8, axis=1)))
    # pbo_fast == pbo_eq1, and block-sum Sharpe == sr_pe
    R2 = rng.normal(0, 1, size=(500, 8))
    i2, o2 = cscv_matrices(R2)
    res["pbo_fast_eq_eq1"] = bool(pbo_fast(i2, o2) == e14.pbo_eq1(i2, o2)[0])
    blocks = np.array_split(np.arange(500), CSCV_S)
    half = list(combinations(range(CSCV_S), 8))[1234]
    rows = np.concatenate([blocks[b] for b in half])
    direct = np.array([e14.sr_pe(R2[rows, k]) for k in range(8)])
    res["blocksum_sr_maxdiff"] = float(np.abs(direct - i2[:, 1234]).max())
    # circular shift keeps the (circular) lag-1 autocorrelation of the labels
    y = (rng.random_sample(900) < 0.4).astype(int)
    y = np.where(rng.random_sample(900) < 0.7, np.roll(y, 1), y)
    ac = lambda v: np.corrcoef(v, np.roll(v, 1))[0, 1]  # noqa: E731
    shifted, _ = permutations_shift(y, 7)
    res["shift_circ_ac1_maxdiff"] = float(max(abs(ac(s) - ac(y)) for s in shifted))
    lin = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]  # noqa: E731
    res["shift_linear_ac1_maxdiff"] = float(max(abs(lin(s) - lin(y)) for s in shifted))
    # block shuffle with a single block is the identity
    y20 = np.arange(20)
    res["block_single_identity"] = bool(all((block_shuffle(y20[:k], np.random.RandomState(i)) == y20[:k]).all()
                                            for k in (1, 15, 20) for i in range(5)))
    res["block_multi_is_permutation"] = bool(np.array_equal(np.sort(block_shuffle(np.arange(95), rng)), np.arange(95)))
    # demeaned returns have mean 0 to 1e-12
    res["demean_abs_mean"] = {a: float(abs(demean(e14.asset_returns(a)[1]).mean())) for a in ("BTC", "ETH", "SOL")}
    # beta = 0 through the drift loop == beta None
    dfa, r = e14.asset_returns("BTC")
    res["beta0_eq_none"] = bool(np.array_equal(e14.make_series(1000, demean(r), len(r), dfa.index, 0.0)[1],
                                               e14.make_series(1000, demean(r), len(r), dfa.index, None)[1]))
    # stationary bootstrap: mean block length close to 20, values from the demeaned vector
    _, rs, idx = make_sb_series(0, demean(r), len(r), dfa.index)
    starts = 1 + int(np.sum(idx[1:] != (idx[:-1] + 1) % len(r)))
    res["sb_mean_block_len"] = len(r) / starts
    print(json.dumps(res, indent=1))
    assert res["cscv_dominant"] == 0.0
    assert res["cscv_identical"] == 1.0
    assert res["pbo_fast_eq_eq1"] and res["blocksum_sr_maxdiff"] < 1e-12
    assert res["shift_circ_ac1_maxdiff"] < 1e-12
    assert res["block_single_identity"] and res["block_multi_is_permutation"]
    assert all(v < 1e-12 for v in res["demean_abs_mean"].values())
    assert res["beta0_eq_none"]
    print("UNIT PASS")


def cmd_repl():
    """S1 and T1 on the non-demeaned E14 base (BTC seeds 0-4) vs results/true_pbo/null.csv."""
    ref = pd.read_csv(os.path.join(HERE, "..", "..", "results", "true_pbo", "null.csv"), keep_default_na=False)
    ref = ref[ref.asset == "BTC"].set_index("seed")
    dfa, r = e14.asset_returns("BTC")
    out, ok = [], True
    for s in range(5):
        df, ret = e14.make_series(s, r, len(r), dfa.index)
        X, y, fwd, split = e14.prepare(df, ret)
        g = gate(X, y, fwd, split, split)
        a = perm_aucs(X, split, split, permutations_full(y, s))
        p95 = float(np.percentile(a, 95))
        pbo = e14.pbo_eq1(*e14.grid_matrices(X, fwd, y, e14.cpcv_paths(len(X), PURGE), GRID_P, make_p))[0]
        rr = ref.loc[s]
        d = {"seed": s, "auc": g["auc"], "auc_ref": float(rr.auc), "perm_p95": p95, "perm_p95_ref": float(rr.perm_p95),
             "pbo_S1": pbo, "pbo_P_ref": float(rr.pbo_P), "net_sr_pe": g["net_sr_pe"], "net_ref": float(rr.net_sr_pe)}
        d["exact"] = bool(d["auc"] == d["auc_ref"] and d["perm_p95"] == d["perm_p95_ref"] and d["pbo_S1"] == d["pbo_P_ref"])
        d["maxdiff"] = max(abs(d["auc"] - d["auc_ref"]), abs(d["perm_p95"] - d["perm_p95_ref"]), abs(d["pbo_S1"] - d["pbo_P_ref"]))
        ok &= d["maxdiff"] < 1e-12
        out.append(d)
        print(json.dumps(d), flush=True)
    print("REPL PASS" if ok else "REPL FAIL")


def run_one(stage, arm, asset, seed):
    p = build(arm, asset, seed)
    if p is None:
        return None, None
    X, y, fwd, split = p
    return (eval_stage1 if stage == 1 else eval_stage2)(X, y, fwd, split, seed)


def cmd_time(stage):
    out = {}
    for arm in STAGE_ARMS[stage]:
        _, _, _, sd = ARMS[arm]
        t = time.time()
        row, _ = run_one(stage, arm, "BTC", seeds_of(sd)[0])
        out[arm] = {"s": round(time.time() - t, 2), "events": row["events"],
                    **{k: round(v, 2) for k, v in row.items() if k.startswith("t_")}}
        print(arm, out[arm], flush=True)
    per = float(np.mean([v["s"] for v in out.values()]))
    proj = per * STAGE_SERIES[stage] / 2 / 3600
    out["summary"] = {"mean_s_per_series": per, "series": STAGE_SERIES[stage], "projected_h_2workers": proj,
                      "prereg_estimate_h": STAGE_EST_H[stage], "limit_h": 2 * STAGE_EST_H[stage],
                      "within_limit": bool(proj <= 2 * STAGE_EST_H[stage])}
    print(json.dumps(out["summary"], indent=1))


def cmd_series(stage, arm, asset, seeds, out):
    rows, npz = [], {}
    t0 = time.time()
    for k, s in enumerate(seeds):
        ts = time.time()
        row, arrs = run_one(stage, arm, asset, s)
        base = {"stage": stage, "arm": arm, "asset": asset, "seed": s, "beta": ARMS[arm][2]}
        if row is None:
            rows.append({**base, "valid": 0})
            continue
        row.update({**base, "valid": 1, "t_total": time.time() - ts})
        rows.append(row)
        for key, v in arrs.items():
            npz[f"{arm}_{asset}_{s}_{key}"] = v
        if (k + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(out + ".csv", index=False)
            np.savez_compressed(out + ".npz", **npz)
            print(f"[s{stage} {arm} {asset}] {k + 1}/{len(seeds)} elapsed {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out + ".csv", index=False)
    np.savez_compressed(out + ".npz", **npz)
    print(f"done {out} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--arm", default="A")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "unit":
        cmd_unit()
    elif a.cmd == "repl":
        cmd_repl()
    elif a.cmd == "time":
        cmd_time(a.stage)
    elif a.cmd == "series":
        assert a.arm in STAGE_ARMS[a.stage]
        assert a.asset == "BTC" or (a.arm == "A" and a.stage == 2)
        sd = a.seeds or ARMS[a.arm][3]
        assert set(seeds_of(sd)) <= set(seeds_of(ARMS[a.arm][3])), "seeds outside the pre-registered range"
        cmd_series(a.stage, a.arm, a.asset, seeds_of(sd), a.out)

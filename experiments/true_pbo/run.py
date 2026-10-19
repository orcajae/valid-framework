"""True-PBO Monte Carlo (CHANGELOG E14), pre-registered in experiments/true_pbo/PREREGISTRATION.md (amendments A1-A4).

Reuses the valid-framework v0.2 pipeline (experiments/mc_expansion.py at 018f4ea) read-only:
series generator, compute_features_simple, CUSUM, Triple Barrier, event-aligned T+1 forward
returns, CPCV N=6 k=2 purge config.CPCV_PURGE, holdout 80/20, base CatBoost model of the AUC gate.
Adds, per series:
  - per-configuration, per-path IS / OOS Sharpe, PER EVENT (mean/sd of position x fwd return; A1)
  - PBO by Eq. 1 (omega_bar = rank/(M+1), count logit(omega_bar) <= 0) for grid P (M=8) and grid R (M=3)
  - cross-check against valid.metrics.compute_pbo on the same matrices
  - Var(SR_IS) by Eq. 2 (valid.metrics.var_sr_is) on grid R (and grid P, display)
  - symmetric permutation test (A3): 100 shuffles, same model/params/iterations as the real fit
  - power arm drift (A2): r'(t+1) = r(t+1) + beta*sigma*z(t), z from r' up to t
Every number is per event; nothing is annualised.

usage (worktree root on sys.path via VF env var, python3 -I):
  run.py unit
  run.py time
  run.py auccheck  --asset BTC --seeds 0-199
  run.py series    --asset BTC --seeds 0-99 --arm null [--beta B] --out PART_PREFIX
  run.py calib     --betas 0,0.05,... --seeds 2000-2019 --out FILE
  run.py real      --out PREFIX
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations

VF = os.environ.get("VF", "/tmp/vf-pbo")
sys.path[:0] = [VF, os.path.join(VF, "experiments")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from experiments import config  # noqa: E402
from experiments.data_sources import load_ohlcv  # noqa: E402
from experiments.mc_expansion import compute_features_simple, get_model_fn  # noqa: E402
from valid.labeling import cusum_filter, triple_barrier_labels  # noqa: E402
from valid.metrics import compute_pbo, var_sr_is  # noqa: E402

COST_RT = config.COST_RETAIL_BP / 10000
N_PERM = 100
EPS = 1e-10

# ---------------------------------------------------------------- grids (fixed by the prereg)
GRID_P = [(d, lr, cw) for d in (4, 6) for lr in (0.03, 0.1) for cw in (None, "Balanced")]  # M = 8
# Hourly real-data arm and grid R: inputs are not distributed with this repository.
# Set PRIVATE_ROOT to a directory with results/round_1h, data/features and results/screening.
PRIVATE_ROOT = os.environ.get("PRIVATE_ROOT", "")
ROUND_1H = os.path.join(PRIVATE_ROOT, "results", "round_1h")
GRID_R_NAMES = ["catboost", "lightgbm", "rf"]  # order as loaded in the private src/paper_extras.py L40-45


def load_grid_r():
    out = []
    if not os.path.isdir(ROUND_1H):  # public checkout: grid R unavailable, grid P and all gates still run
        return out
    for mt in GRID_R_NAMES:
        with open(os.path.join(ROUND_1H, f"optuna_{mt}.json")) as f:
            out.append((mt, json.load(f)["params"]))
    return out


GRID_R = load_grid_r()


def make_p(cfg):
    from catboost import CatBoostClassifier
    d, lr, cw = cfg
    kw = {"auto_class_weights": cw} if cw else {}
    return CatBoostClassifier(depth=d, iterations=100, learning_rate=lr, verbose=0, random_seed=42, **kw)


def make_r(cfg):
    """build_model of src/paper_extras.py L95-107; RF n_jobs=1 (parallel cap; result unchanged)."""
    mt, p = cfg
    if mt == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(depth=p["depth"], iterations=p["iterations"], learning_rate=p["lr"],
                                  l2_leaf_reg=p["l2"], auto_class_weights="Balanced", verbose=0, random_seed=42)
    if mt == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(max_depth=p["depth"], n_estimators=p["n_est"], learning_rate=p["lr"],
                              reg_lambda=p["reg"], class_weight="balanced", verbose=-1, random_state=42, n_jobs=1)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(max_depth=p["depth"], n_estimators=p["n_est"], min_samples_leaf=p["min_leaf"],
                                  class_weight="balanced", random_state=42, n_jobs=1)


# ---------------------------------------------------------------- statistics
def sr_pe(x):
    """Per-event Sharpe (A1): mean / sd, same epsilon as mc_expansion, no annualisation."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return x.mean() / (x.std() + EPS)


def pbo_eq1(is_m, oos_m):
    """Manuscript Eq. 1: omega = OOS rank (1 = worst, M = best, ties averaged) of the IS-best
    configuration, omega_bar = omega/(M+1), PBO = share of paths with logit(omega_bar) <= 0."""
    cnt = tot = 0
    for j in range(is_m.shape[1]):
        a, b = is_m[:, j], oos_m[:, j]
        ok = ~(np.isnan(a) | np.isnan(b))
        M = int(ok.sum())
        if M < 2:
            continue
        best = int(np.argmax(a[ok]))
        wbar = rankdata(b[ok])[best] / (M + 1)
        cnt += int(np.log(wbar / (1 - wbar)) <= 0)
        tot += 1
    return (cnt / tot if tot else np.nan), cnt, tot


# ---------------------------------------------------------------- series
def make_series(seed, ret_actual, n_days, ref_index, beta=None):
    """mc_expansion.run_mc_iteration L127-139, with the A2 drift inserted after the bootstrap draw
    (same RNG call order, so beta=None reproduces the null series exactly)."""
    rng = np.random.RandomState(seed)
    r = rng.choice(ret_actual, size=n_days, replace=True)
    if beta is not None:
        mu, sig = r.mean(), r.std()
        rp = r.copy()
        for t in range(n_days - 1):
            z = 0.0 if t < 4 else (rp[t - 4:t + 1].sum() - 5 * mu) / (sig * np.sqrt(5))
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


def prepare(df, ret):
    """mc_expansion L141-170: features, CUSUM, TB labels, X/y, event-aligned fwd returns."""
    feats = compute_features_simple(df)
    c = df["close"]
    vol = c.pct_change().ewm(span=60).std()
    thr = vol.mean() if vol.mean() > 0 else 0.02
    events = cusum_filter(c, thr)
    tb = triple_barrier_labels(df, events, pt_mult=2.0, sl_mult=2.5, max_hold=20, vol_window=60)
    feats["tb_label"] = tb
    data = feats.dropna(subset=["tb_label"])
    data = data[data["tb_label"].isin([-1, 1])].copy()
    feat_cols = [col for col in data.columns if col != "tb_label"]
    usable = [f for f in feat_cols if data[f].isna().mean() < 0.5]
    if len(data) < 100 or len(usable) < 5:
        return None
    X = data[usable].ffill().fillna(0).values
    y = (data["tb_label"] == 1).astype(int).values
    split = int(len(X) * 0.8)
    if split < 30 or len(X) - split < 10:
        return None
    bar_idx = df.index.get_indexer(data.index)
    fwd_bar = np.append(np.asarray(ret)[1:], 0.0)
    return X, y, fwd_bar[bar_idx], split


def cpcv_paths(n_s, purge):
    """mc_expansion L180-198 (N=6, k=2, purge on both sides, no embargo)."""
    gs = n_s // 6
    gids = np.zeros(n_s, dtype=int)
    for g in range(6):
        gids[g * gs:(g + 1) * gs if g < 5 else n_s] = g
    out = []
    for tg in combinations(range(6), 2):
        test_mask = np.isin(gids, tg)
        tr_i, te_i = np.where(~test_mask)[0], np.where(test_mask)[0]
        pm = np.zeros(n_s, dtype=bool)
        for t in sorted(tg):
            ti = np.where(gids == t)[0]
            s2, e2 = ti[0], ti[-1]
            pm[max(0, s2 - purge):s2] = True
            pm[e2 + 1:min(n_s, e2 + purge + 1)] = True
        out.append((tr_i[~pm[tr_i]], te_i))
    return out


def grid_matrices(X, fwd, y, paths, grid, maker, min_te=5, min_tr=10):
    M, J = len(grid), len(paths)
    is_m, oos_m = np.full((M, J), np.nan), np.full((M, J), np.nan)
    for j, (tr_i, te_i) in enumerate(paths):
        if len(te_i) < min_te or len(tr_i) < min_tr:
            continue
        for ci, cfg in enumerate(grid):
            m = maker(cfg)
            m.fit(X[tr_i], y[tr_i])
            p_is = np.asarray(m.predict(X[tr_i])).ravel()
            p_te = np.asarray(m.predict(X[te_i])).ravel()
            is_m[ci, j] = sr_pe(np.where(p_is == 1, 1.0, -1.0) * fwd[tr_i])
            oos_m[ci, j] = sr_pe(np.where(p_te == 1, 1.0, -1.0) * fwd[te_i])
    return is_m, oos_m


def base_gate(X, y, fwd, split, seed, n_perm):
    make, name = get_model_fn("catboost", True, iterations=100)
    m = make()
    m.fit(X[:split], y[:split])
    try:
        auc = roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1])
    except Exception:
        auc = 0.5
    preds = np.asarray(m.predict(X[split:])).ravel()
    pos = np.where(preds == 1, 1.0, -1.0)
    strat = pos * fwd[split:]
    trades = np.abs(np.diff(np.concatenate([[0], preds]))).sum()
    cost = trades * COST_RT / len(preds)
    out = {"auc": auc, "net_sr_pe": (strat.mean() - cost) / (strat.std() + EPS),
           "gross_sr_pe": strat.mean() / (strat.std() + EPS), "long_pct": float((preds == 1).mean())}
    perm = []
    for pi in range(n_perm):  # A3: same model, params, iterations as the real fit
        ys = np.random.RandomState(seed * 1000 + pi).permutation(y)
        pm = make()
        pm.fit(X[:split], ys[:split])
        try:
            perm.append(roc_auc_score(ys[split:], pm.predict_proba(X[split:])[:, 1]))
        except Exception:
            perm.append(0.5)
    perm = np.array(perm)
    if n_perm:
        out["perm_p95"] = float(np.percentile(perm, 95))
        out["perm_pass"] = bool(auc > out["perm_p95"])
    return out, perm


def evaluate(X, y, fwd, split, seed, purge, n_perm=N_PERM, grids=("P", "R")):
    row, arrs = {"events": len(X)}, {}
    t0 = time.time()
    g, perm = base_gate(X, y, fwd, split, seed, n_perm)
    row.update(g)
    arrs["perm_aucs"] = perm
    row["t_base"] = time.time() - t0
    paths = cpcv_paths(len(X), purge)
    for gname in [g for g in grids if g != "R" or GRID_R]:  # grid R needs the private inputs
        t1 = time.time()
        grid, maker = (GRID_P, make_p) if gname == "P" else (GRID_R, make_r)
        is_m, oos_m = grid_matrices(X, fwd, y, paths, grid, maker)
        arrs[f"is_{gname}"], arrs[f"oos_{gname}"] = is_m, oos_m
        p1, c1, n1 = pbo_eq1(is_m, oos_m)
        p2, _, _ = compute_pbo(is_m, oos_m)
        row[f"pbo_{gname}"] = p1
        row[f"pbo_{gname}_cmp"] = p2
        row[f"pbo_{gname}_disagree"] = int(not (np.isclose(p1, p2) or (np.isnan(p1) and np.isnan(p2))))
        row[f"paths_{gname}"] = n1
        row[f"var_sr_is_{gname}"] = var_sr_is(is_m)
        row[f"t_{gname}"] = time.time() - t1
    return row, arrs


# ---------------------------------------------------------------- data
def asset_returns(asset):
    """mc_expansion.run_setting L262-268."""
    df = load_ohlcv("real", asset)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    r = df["close"].pct_change().dropna().values
    return df, r


def seeds_of(s):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


# ---------------------------------------------------------------- commands
def cmd_unit():
    rng = np.random.RandomState(0)
    M, J = 8, 15
    is_m = rng.normal(size=(M, J))
    best = is_m.argmax(0)
    good, bad = rng.normal(size=(M, J)), rng.normal(size=(M, J))
    for j in range(J):
        good[best[j], j] = good[:, j].max() + 1.0   # IS-best is OOS-best
        bad[best[j], j] = bad[:, j].min() - 1.0     # IS-best is OOS-worst
    res = {"pbo_best_is_best": pbo_eq1(is_m, good)[0], "pbo_best_is_worst": pbo_eq1(is_m, bad)[0],
           "cmp_best_is_best": compute_pbo(is_m, good)[0], "cmp_best_is_worst": compute_pbo(is_m, bad)[0]}
    # M = 3, IS-best is OOS-middle: Eq. 1 counts it (2/4 = 0.5), compute_pbo does not (2/3 > 0.5)
    i3 = np.array([[3.0], [2.0], [1.0]])
    o3 = np.array([[2.0], [3.0], [1.0]])
    res["M3_middle_eq1"], res["M3_middle_cmp"] = pbo_eq1(i3, o3)[0], compute_pbo(i3, o3)[0]
    print(json.dumps(res, indent=1))
    assert res["pbo_best_is_best"] == 0.0 and res["pbo_best_is_worst"] == 1.0
    assert res["cmp_best_is_best"] == 0.0 and res["cmp_best_is_worst"] == 1.0
    print("UNIT PASS")


def cmd_time():
    dfa, r = asset_returns("BTC")
    df, ret = make_series(0, r, len(r), dfa.index)
    X, y, fwd, split = prepare(df, ret)
    out = {}
    t = time.time(); base_gate(X, y, fwd, split, 0, N_PERM); out["base+perm100"] = time.time() - t
    paths = cpcv_paths(len(X), config.CPCV_PURGE)
    t = time.time(); grid_matrices(X, fwd, y, paths, GRID_P, make_p); out["grid_P"] = time.time() - t
    t = time.time(); grid_matrices(X, fwd, y, paths, GRID_R, make_r); out["grid_R"] = time.time() - t
    per = sum(out.values())
    n_series = 600 + 200
    out.update({"events": len(X), "per_series_s": per, "series_total": n_series,
                "projected_h_parallel2": per * n_series / 2 / 3600, "projected_h_serial": per * n_series / 3600})
    print(json.dumps(out, indent=1))


def cmd_auccheck(asset, seeds):
    dfa, r = asset_returns(asset)
    make, _ = get_model_fn("catboost", True, iterations=100)
    aucs = []
    for s in seeds:
        df, ret = make_series(s, r, len(r), dfa.index)
        p = prepare(df, ret)
        if p is None:
            continue
        X, y, _, split = p
        m = make(); m.fit(X[:split], y[:split])
        aucs.append(roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1]))
    aucs = np.array(aucs)
    print(json.dumps({"asset": asset, "n": len(aucs), "auc_gt_055_pct": round(100 * (aucs > 0.55).mean(), 2),
                      "mean_auc": round(aucs.mean(), 4), "sd_auc": round(aucs.std(ddof=1), 4)}))


def cmd_series(asset, seeds, arm, beta, out):
    dfa, r = asset_returns(asset)
    rows, npz = [], {}
    t0 = time.time()
    for k, s in enumerate(seeds):
        ts = time.time()
        df, ret = make_series(s, r, len(r), dfa.index, beta)
        p = prepare(df, ret)
        if p is None:
            rows.append({"arm": arm, "asset": asset, "seed": s, "beta": beta, "valid": 0})
            continue
        X, y, fwd, split = p
        row, arrs = evaluate(X, y, fwd, split, s, config.CPCV_PURGE)
        row.update({"arm": arm, "asset": asset, "seed": s, "beta": beta, "valid": 1, "t_total": time.time() - ts})
        rows.append(row)
        for key, v in arrs.items():
            npz[f"{asset}_{s}_{key}"] = v
        if (k + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(out + ".csv", index=False)
            np.savez_compressed(out + ".npz", **npz)
            print(f"[{asset} {arm} beta={beta}] {k + 1}/{len(seeds)} elapsed {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out + ".csv", index=False)
    np.savez_compressed(out + ".npz", **npz)
    print(f"done {out} {time.time() - t0:.0f}s", flush=True)


def cmd_calib(betas, seeds, out):
    dfa, r = asset_returns("BTC")
    make, _ = get_model_fn("catboost", True, iterations=100)
    rows = []
    for b in betas:
        aucs = []
        for s in seeds:
            df, ret = make_series(s, r, len(r), dfa.index, b)
            p = prepare(df, ret)
            if p is None:
                continue
            X, y, _, split = p
            m = make(); m.fit(X[:split], y[:split])
            aucs.append(roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1]))
        rows.append({"beta": b, "n": len(aucs), "mean_auc": float(np.mean(aucs)), "sd_auc": float(np.std(aucs, ddof=1))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def cmd_real(out):
    rows, npz = [], {}
    # --- BTC daily, actual series (no bootstrap), same pipeline as the MC
    df, r = asset_returns("BTC")
    ret = df["close"].pct_change().fillna(0).values
    dfr = df[["open", "high", "low", "close", "volume"]].copy()
    X, y, fwd, split = prepare(dfr, ret)
    row, arrs = evaluate(X, y, fwd, split, 0, config.CPCV_PURGE, n_perm=N_PERM, grids=("P", "R"))
    row.update({"arm": "real", "asset": "BTC", "setting": "btc_daily_actual"})
    rows.append(row)
    npz.update({f"btc_daily_{k}": v for k, v in arrs.items()})
    print(row, flush=True)
    # --- round_1h pipeline (private src/paper_extras.py L36-90): data, features, purge 24 + embargo 12, fwd_ret_4
    feat = os.path.join(PRIVATE_ROOT, "data", "features", "features_with_targets_1h.parquet")
    d1 = pd.read_parquet(feat)
    data = d1[d1["tb_label"].notna()]
    data = data[data["tb_label"].isin([-1, 1])].copy()
    feats = pd.read_csv(os.path.join(PRIVATE_ROOT, "results", "screening", "1h_shap_ranking.csv"))["feature"].tolist()
    usable = [f for f in feats if f in data.columns and data[f].isna().mean() < 0.40]
    X1 = data[usable].ffill().fillna(data[usable].median()).values
    y1 = (data["tb_label"] == 1).astype(int).values
    f1 = data["fwd_ret_4"].fillna(0).values
    n = len(X1)
    gs = n // 6
    gids = np.zeros(n, dtype=int)
    for g in range(6):
        gids[g * gs:(g + 1) * gs if g < 5 else n] = g
    paths = []
    for tg in combinations(range(6), 2):
        te = np.where(np.isin(gids, tg))[0]
        tr = np.where(~np.isin(gids, tg))[0]
        pm = np.zeros(n, dtype=bool)
        for t in sorted(tg):
            ti = np.where(gids == t)[0]
            s, e = ti[0], ti[-1]
            pm[max(0, s - 24):s] = True
            pm[e + 1:min(n, e + 25)] = True
            pm[max(0, s - 36):max(0, s - 24)] = True
            pm[min(n, e + 25):min(n, e + 37)] = True
        paths.append((tr[~pm[tr]], te))
    for gname, grid, maker in (("R", GRID_R, make_r), ("P", GRID_P, make_p)):
        t1 = time.time()
        is_m, oos_m = grid_matrices(X1, f1, y1, paths, grid, maker, min_te=10, min_tr=30)
        npz[f"round_1h_is_{gname}"], npz[f"round_1h_oos_{gname}"] = is_m, oos_m
        p1, c1, n1 = pbo_eq1(is_m, oos_m)
        rr = {"arm": "real", "asset": "BTC", "setting": f"round_1h_grid{gname}", "events": n,
              f"pbo_{gname}": p1, f"pbo_{gname}_cmp": compute_pbo(is_m, oos_m)[0], f"paths_{gname}": n1,
              f"var_sr_is_{gname}": var_sr_is(is_m),
              f"var_sr_is_{gname}_x252_check": var_sr_is(is_m * np.sqrt(252)), f"t_{gname}": time.time() - t1}
        rows.append(rr)
        print(rr, flush=True)
    pd.DataFrame(rows).to_csv(out + ".csv", index=False)
    np.savez_compressed(out + ".npz", **npz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--arm", default="null")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--betas", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "unit":
        cmd_unit()
    elif a.cmd == "time":
        cmd_time()
    elif a.cmd == "auccheck":
        cmd_auccheck(a.asset, seeds_of(a.seeds))
    elif a.cmd == "series":
        cmd_series(a.asset, seeds_of(a.seeds), a.arm, a.beta, a.out)
    elif a.cmd == "calib":
        cmd_calib([float(b) for b in a.betas.split(",")], seeds_of(a.seeds), a.out)
    elif a.cmd == "real":
        cmd_real(a.out)

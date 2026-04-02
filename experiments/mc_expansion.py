"""
MC Expansion: 3 additional settings (ETH 1h, SOL 1h, BTC daily)
Each: 200 iterations, seeds 0-199, identical pipeline to BTC 1h.

Usage:
  python mc_expansion.py --setting eth_1h   # ETH 1h CatBoost balanced
  python mc_expansion.py --setting sol_1h   # SOL 1h CatBoost balanced
  python mc_expansion.py --setting btc_daily # BTC daily CatBoost balanced
  python mc_expansion.py --all              # Run all 3 sequentially
"""
import os, sys, warnings, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

RAW = Path(os.path.expanduser("~/jwquant/data/raw"))
OUT = Path(os.path.expanduser("~/jwquant/results/paper_kbs"))
OUT.mkdir(parents=True, exist_ok=True)

COST_RT = 18 / 10000
N_MC = 200
N_PERM = 20  # permutation shuffles per iteration

# ═══ Settings ═══
SETTINGS = {
    "eth_1h": {
        "file": "binance_ethusdt_ohlcv_1d.parquet",
        "label": "ETH 1h",
        "out_name": "mc_fpr_eth_1h_200.csv",
    },
    "sol_1h": {
        "file": "binance_solusdt_ohlcv_1d.parquet",
        "label": "SOL 1h",
        "out_name": "mc_fpr_sol_1h_200.csv",
    },
    "btc_daily": {
        "file": "binance_btcusdt_ohlcv_1d.parquet",
        "label": "BTC daily",
        "out_name": "mc_fpr_btc_daily_200.csv",
    },
}


def sr_f(r):
    return r.mean() / r.std() * np.sqrt(252) if len(r) > 1 and r.std() > 0 else 0.0


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    adj = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0, centre - adj), min(1, centre + adj)


def compute_features_simple(df):
    c = df["close"]
    v = df["volume"]
    ret1 = c.pct_change()
    feats = pd.DataFrame(index=df.index)
    for p in [1, 5, 10, 20, 60, 120]:
        feats[f"ret_{p}"] = c.pct_change(p)
    for p in [14, 30, 60]:
        feats[f"vol_{p}"] = ret1.rolling(p).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feats["macd_norm"] = (ema12 - ema26) / ema26.replace(0, np.nan)
    feats["macd_hist"] = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()) / ema26.replace(0, np.nan)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feats["bb_pos"] = (c - sma20) / (2 * std20).replace(0, np.nan)
    feats["bb_width"] = (4 * std20) / sma20.replace(0, np.nan)
    feats["vol_z_14"] = (v - v.rolling(14).mean()) / v.rolling(14).std().replace(0, np.nan)
    feats["range_pct"] = ((df["high"] - df["low"]) / c).rolling(14).mean()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - c.shift(1)).abs(), (df["low"] - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    feats["atr_norm"] = tr.rolling(14).mean() / c
    return feats


def cusum_filter(prices, threshold):
    events = []
    s_pos = s_neg = 0.0
    ret = prices.pct_change().dropna()
    for i in range(len(ret)):
        s_pos = max(0, s_pos + ret.iloc[i])
        s_neg = min(0, s_neg + ret.iloc[i])
        if s_pos > threshold:
            events.append(ret.index[i])
            s_pos = 0
        elif s_neg < -threshold:
            events.append(ret.index[i])
            s_neg = 0
    return pd.DatetimeIndex(events)


def triple_barrier(df, events, pt=2.0, sl=2.5, max_hold=20, vol_win=60):
    c = df["close"]
    vol = c.pct_change().ewm(span=vol_win).std()
    labels = {}
    for t in events:
        if t not in c.index:
            continue
        loc = c.index.get_loc(t)
        if loc + 1 >= len(c):
            continue
        ep = c.iloc[loc]
        v = vol.iloc[loc]
        if pd.isna(v) or v == 0:
            continue
        pt_b = ep * (1 + pt * v)
        sl_b = ep * (1 - sl * v)
        end = min(loc + max_hold, len(c) - 1)
        label = 0
        for j in range(loc + 1, end + 1):
            if c.iloc[j] >= pt_b:
                label = 1
                break
            elif c.iloc[j] <= sl_b:
                label = -1
                break
        labels[t] = label
    return pd.Series(labels, name="tb_label")


def run_mc_iteration(rng_seed, ret_actual, n_days, ref_index, balanced=True):
    """Run one MC iteration with given asset's return distribution."""
    rng = np.random.RandomState(rng_seed)
    synth_ret = rng.choice(ret_actual, size=n_days, replace=True)
    synth_price = 10000 * np.exp(np.cumsum(synth_ret))
    synth_df = pd.DataFrame(
        {
            "open": synth_price,
            "high": synth_price * (1 + np.abs(rng.normal(0, 0.01, n_days))),
            "low": synth_price * (1 - np.abs(rng.normal(0, 0.01, n_days))),
            "close": synth_price,
            "volume": rng.lognormal(20, 1, n_days),
        },
        index=ref_index[:n_days],
    )

    feats = compute_features_simple(synth_df)
    c = synth_df["close"]
    vol = c.pct_change().ewm(span=60).std()
    thr = vol.mean() if vol.mean() > 0 else 0.02
    events = cusum_filter(c, thr)
    tb = triple_barrier(synth_df, events)
    feats["tb_label"] = tb
    data = feats.dropna(subset=["tb_label"])
    mask = data["tb_label"].isin([-1, 1])
    data = data[mask].copy()
    feat_cols = [col for col in data.columns if col != "tb_label"]
    usable = [f for f in feat_cols if data[f].isna().mean() < 0.5]

    if len(data) < 100 or len(usable) < 5:
        return None

    X = data[usable].ffill().fillna(0).values
    y = (data["tb_label"] == 1).astype(int).values
    split = int(len(X) * 0.8)
    if split < 30 or len(X) - split < 10:
        return None

    kw = {"auto_class_weights": "Balanced"} if balanced else {}
    m = CatBoostClassifier(depth=5, iterations=100, learning_rate=0.1, verbose=0, random_seed=42, **kw)
    m.fit(X[:split], y[:split])
    try:
        auc = roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1])
    except Exception:
        auc = 0.5

    # CPCV PBO
    n_s = len(X)
    gs = n_s // 6
    gids = np.zeros(n_s, dtype=int)
    for g in range(6):
        s = g * gs
        e = (g + 1) * gs if g < 5 else n_s
        gids[s:e] = g
    combos_mc = list(combinations(range(6), 2))

    is_accs = []
    oos_accs = []
    is_srs_list = []
    for tg in combos_mc:
        test_mask = np.isin(gids, tg)
        tr_i = np.where(~test_mask)[0]
        te_i = np.where(test_mask)[0]
        pm = np.zeros(n_s, dtype=bool)
        for t in sorted(tg):
            ti = np.where(gids == t)[0]
            s2, e2 = ti[0], ti[-1]
            pm[max(0, s2 - 20) : s2] = True
            pm[e2 + 1 : min(n_s, e2 + 21)] = True
        tr_i = tr_i[~pm[tr_i]]
        if len(te_i) < 5 or len(tr_i) < 10:
            continue
        try:
            mc_m = CatBoostClassifier(depth=5, iterations=100, learning_rate=0.1, verbose=0, random_seed=42, **kw)
            mc_m.fit(X[tr_i], y[tr_i])
            is_a = np.mean(mc_m.predict(X[tr_i]) == y[tr_i])
            oos_a = np.mean(mc_m.predict(X[te_i]) == y[te_i])
            is_accs.append(is_a)
            oos_accs.append(oos_a)
            preds_is = mc_m.predict(X[tr_i])
            pos_is = np.where(preds_is == 1, 1.0, -1.0)
            fwd_is = synth_ret[:n_s][tr_i] if len(synth_ret) >= n_s else np.zeros(len(tr_i))
            sr_is = fwd_is[: len(pos_is)].mean() / (fwd_is[: len(pos_is)].std() + 1e-10) * np.sqrt(252)
            is_srs_list.append(sr_is)
        except Exception:
            pass

    pbo = np.nan
    var_sr_is = np.nan
    if len(is_accs) > 3:
        pbo_count = sum(1 for i, o in zip(is_accs, oos_accs) if i > o)
        pbo = pbo_count / len(is_accs)
    if len(is_srs_list) > 2:
        var_sr_is = np.var(is_srs_list)

    # Net SR
    preds = m.predict(X[split:])
    pos = np.where(preds == 1, 1.0, -1.0)
    synth_fwd = synth_ret[split : split + len(preds)]
    if len(synth_fwd) >= len(preds):
        synth_fwd = synth_fwd[: len(preds)]
        strat = synth_fwd * pos
        trades = np.abs(np.diff(np.concatenate([[0], preds]))).sum()
        cost_daily = (trades * COST_RT) / len(preds) if len(preds) > 0 else 0
        net_sr = (strat.mean() - cost_daily) / (strat.std() + 1e-10) * np.sqrt(252)
        gross_sr = strat.mean() / (strat.std() + 1e-10) * np.sqrt(252)
    else:
        net_sr = np.nan
        gross_sr = np.nan

    long_pct = (preds == 1).mean()

    # Permutation test
    perm_passed = False
    perm_aucs = []
    for pi in range(N_PERM):
        prng = np.random.RandomState(rng_seed * 1000 + pi)
        ys = prng.permutation(y)
        pm_m = CatBoostClassifier(depth=5, iterations=50, learning_rate=0.1, verbose=0, random_seed=42, **kw)
        pm_m.fit(X[:split], ys[:split])
        try:
            pa = roc_auc_score(ys[split:], pm_m.predict_proba(X[split:])[:, 1])
        except Exception:
            pa = 0.5
        perm_aucs.append(pa)
    perm_p95 = np.percentile(perm_aucs, 95)
    perm_passed = auc > perm_p95

    return {
        "auc": auc,
        "pbo": pbo,
        "net_sr": net_sr,
        "gross_sr": gross_sr,
        "var_sr_is": var_sr_is,
        "long_pct": long_pct,
        "perm_passed": perm_passed,
        "events": len(data),
    }


def run_setting(setting_key):
    cfg = SETTINGS[setting_key]
    print(f"\n{'='*70}")
    print(f"MC EXPANSION: {cfg['label']} — {N_MC} iterations")
    print(f"{'='*70}")

    # Load data
    df = pd.read_parquet(RAW / cfg["file"])
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    ret_actual = df["close"].pct_change().dropna().values
    n_days = len(ret_actual)
    ref_index = df.index

    print(f"  Source: {cfg['file']} — {n_days} bars ({df.index[0].date()} to {df.index[-1].date()})")

    results = []
    t0 = time.time()
    for seed in range(N_MC):
        res = run_mc_iteration(seed, ret_actual, n_days, ref_index, balanced=True)
        if res is not None:
            res["mc_iter"] = seed
            res["setting"] = setting_key
            results.append(res)
        if (seed + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (seed + 1) * (N_MC - seed - 1)
            valid = len(results)
            print(f"  [{seed+1:3d}/{N_MC}] valid={valid} elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    df_res = pd.DataFrame(results)
    out_path = OUT / cfg["out_name"]
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(df_res)} valid iterations)")

    # Summary
    n = len(df_res)
    if n > 0:
        fpr_auc = (df_res["auc"] > 0.55).mean()
        fpr_pbo = (df_res["pbo"] < 0.20).mean()
        fpr_sr = (df_res["net_sr"] > 0).mean()
        fpr_perm = df_res["perm_passed"].mean()
        fpr_valid = ((df_res["auc"] > 0.55) & (df_res["pbo"] < 0.20) & (df_res["net_sr"] > 0)).mean()

        print(f"\n  === {cfg['label']} FPR Summary (n={n}) ===")
        for name, fpr in [
            ("AUC > 0.55", fpr_auc),
            ("Permutation", fpr_perm),
            ("PBO < 0.20", fpr_pbo),
            ("Net SR > 0", fpr_sr),
            ("Full VALID", fpr_valid),
        ]:
            lo, hi = wilson_ci(fpr, n)
            print(f"  {name:20s}: {fpr*100:5.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")

        mean_auc = df_res["auc"].mean()
        std_auc = df_res["auc"].std()
        var_sr = df_res["var_sr_is"].dropna()
        print(f"\n  Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        if len(var_sr) > 0:
            print(f"  Var(SR_IS): mean={var_sr.mean():.3f}, 95th={var_sr.quantile(0.95):.3f}")

    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=list(SETTINGS.keys()), help="Which setting to run")
    parser.add_argument("--all", action="store_true", help="Run all 3 settings")
    args = parser.parse_args()

    if args.all:
        all_results = {}
        for key in SETTINGS:
            all_results[key] = run_setting(key)

        # Combined summary
        print(f"\n{'='*70}")
        print("COMBINED SUMMARY — All 4 Settings")
        print(f"{'='*70}")

        # Load existing BTC 1h
        btc_1h = pd.read_csv(OUT / "monte_carlo_fpr_200.csv")
        btc_1h["setting"] = "btc_1h"

        combined = pd.concat([btc_1h] + list(all_results.values()), ignore_index=True)
        combined.to_csv(OUT / "mc_fpr_all_settings_200.csv", index=False)

        for setting in ["btc_1h", "eth_1h", "sol_1h", "btc_daily"]:
            sub = combined[combined["setting"] == setting]
            n = len(sub)
            if n == 0:
                continue
            fpr_auc = (sub["auc"] > 0.55).mean()
            fpr_pbo = (sub["pbo"] < 0.20).mean()
            lo_a, hi_a = wilson_ci(fpr_auc, n)
            lo_p, hi_p = wilson_ci(fpr_pbo, n)
            print(f"  {setting:12s} (n={n:3d}): AUC FPR={fpr_auc*100:5.1f}% [{lo_a*100:.1f}%,{hi_a*100:.1f}%]  PBO FPR={fpr_pbo*100:5.1f}% [{lo_p*100:.1f}%,{hi_p*100:.1f}%]")

        print(f"\n  Saved: {OUT / 'mc_fpr_all_settings_200.csv'}")

    elif args.setting:
        run_setting(args.setting)
    else:
        parser.print_help()

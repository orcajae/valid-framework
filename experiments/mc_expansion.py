"""Monte Carlo null-pipeline FPR experiment.

For each setting, resample the asset's daily returns i.i.d. (a synthetic
series with NO exploitable signal), run the full ML evaluation pipeline on
it, and measure how often each evaluation gate wrongly declares success.
200 iterations per setting (seeds 0-199) in the reference runs.

Settings are daily-bar experiments (btc_daily / eth_daily / sol_daily).
Historical note: reference CSVs named mc_fpr_eth_1h_200.csv /
mc_fpr_sol_1h_200.csv were produced by this pipeline on DAILY bars under
mislabeled setting keys; see REPRODUCE.md for the mapping.

v0.2 deviations from the v0.1 reference run (documented in REPRODUCE.md):
- Var(SR_IS) and holdout SR now apply the model's positions to
  event-aligned T+1 forward returns. v0.1 computed them from raw resampled
  returns at misaligned (event-row) indices and never used the positions.
- Data source is configurable (deterministic synthetic default, --data real
  for public Binance CSVs); model registry falls back to sklearn
  HistGradientBoosting when CatBoost is unavailable (--model to force).

Usage:
  python experiments/mc_expansion.py --all [--data synthetic|real]
                                     [--model auto|catboost|hgb]
                                     [--n-mc 200] [--smoke]
"""
import argparse
import time
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from experiments import config
    from experiments.data_sources import load_ohlcv
except ImportError:
    import config
    from data_sources import load_ohlcv

from sklearn.metrics import roc_auc_score

from valid.labeling import cusum_filter, triple_barrier_labels
from valid.metrics import wilson_ci

COST_RT = config.COST_RETAIL_BP / 10000

SETTINGS = {
    "btc_daily": {"asset": "BTC", "label": "BTC daily"},
    "eth_daily": {"asset": "ETH", "label": "ETH daily"},
    "sol_daily": {"asset": "SOL", "label": "SOL daily"},
}


def get_model_fn(name="auto", balanced=True, iterations=100):
    """Model registry. Reference runs used CatBoost; the sklearn
    HistGradientBoosting fallback keeps the pipeline runnable with core
    dependencies only (CI). Returns (factory, resolved_name)."""
    if name in ("auto", "catboost"):
        try:
            from catboost import CatBoostClassifier

            kw = {"auto_class_weights": "Balanced"} if balanced else {}

            def make():
                return CatBoostClassifier(
                    depth=5, iterations=iterations, learning_rate=0.1,
                    verbose=0, random_seed=42, **kw)

            return make, "catboost"
        except ImportError:
            if name == "catboost":
                raise
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make():
        return HistGradientBoostingClassifier(
            max_depth=5, max_iter=iterations, learning_rate=0.1,
            random_state=42, class_weight="balanced" if balanced else None)

    return make, "hgb"


def compute_features_simple(df):
    """Frozen reference-pipeline feature set (do not swap for
    valid.features — a different feature set would silently change the
    null-pipeline results)."""
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


def run_mc_iteration(rng_seed, ret_actual, n_days, ref_index, model="auto",
                     balanced=True, n_perm=20):
    """Run one MC iteration with the given asset's return distribution."""
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
    tb = triple_barrier_labels(synth_df, events, pt_mult=2.0, sl_mult=2.5,
                               max_hold=20, vol_window=60)
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
    n_s = len(X)

    # Event-row -> bar-position map and T+1 forward return per event
    # (v0.2 fix: v0.1 indexed raw bar returns with event-row indices).
    bar_idx = synth_df.index.get_indexer(data.index)
    fwd_bar = np.append(synth_ret[1:], 0.0)
    fwd_evt = fwd_bar[bar_idx]

    make_model, model_name = get_model_fn(model, balanced, iterations=100)
    m = make_model()
    m.fit(X[:split], y[:split])
    try:
        auc = roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1])
    except Exception:
        auc = 0.5

    # CPCV PBO + Var(SR_IS)
    gs = n_s // 6
    gids = np.zeros(n_s, dtype=int)
    for g in range(6):
        s = g * gs
        e = (g + 1) * gs if g < 5 else n_s
        gids[s:e] = g

    is_accs, oos_accs, is_srs_list = [], [], []
    for tg in combinations(range(6), 2):
        test_mask = np.isin(gids, tg)
        tr_i = np.where(~test_mask)[0]
        te_i = np.where(test_mask)[0]
        pm = np.zeros(n_s, dtype=bool)
        for t in sorted(tg):
            ti = np.where(gids == t)[0]
            s2, e2 = ti[0], ti[-1]
            pm[max(0, s2 - config.CPCV_PURGE):s2] = True
            pm[e2 + 1:min(n_s, e2 + config.CPCV_PURGE + 1)] = True
        tr_i = tr_i[~pm[tr_i]]
        if len(te_i) < 5 or len(tr_i) < 10:
            continue
        try:
            mc_m = make_model()
            mc_m.fit(X[tr_i], y[tr_i])
            preds_is = np.asarray(mc_m.predict(X[tr_i])).ravel()
            is_accs.append(np.mean(preds_is == y[tr_i]))
            oos_accs.append(np.mean(np.asarray(mc_m.predict(X[te_i])).ravel() == y[te_i]))
            # v0.2: strategy IS SR = positions x event-aligned fwd returns
            pos_is = np.where(preds_is == 1, 1.0, -1.0)
            strat_is = pos_is * fwd_evt[tr_i]
            sr_is = strat_is.mean() / (strat_is.std() + 1e-10) * np.sqrt(252)
            is_srs_list.append(sr_is)
        except Exception:
            pass

    pbo = np.nan
    var_sr_is = np.nan
    if len(is_accs) > 3:
        pbo = sum(1 for i, o in zip(is_accs, oos_accs) if i > o) / len(is_accs)
    if len(is_srs_list) > 2:
        var_sr_is = np.var(is_srs_list)

    # Holdout net SR (v0.2: event-aligned forward returns)
    preds = np.asarray(m.predict(X[split:])).ravel()
    pos = np.where(preds == 1, 1.0, -1.0)
    strat = pos * fwd_evt[split:]
    trades = np.abs(np.diff(np.concatenate([[0], preds]))).sum()
    cost_daily = (trades * COST_RT) / len(preds) if len(preds) > 0 else 0
    net_sr = (strat.mean() - cost_daily) / (strat.std() + 1e-10) * np.sqrt(252)
    gross_sr = strat.mean() / (strat.std() + 1e-10) * np.sqrt(252)
    long_pct = (preds == 1).mean()

    # Permutation test
    perm_aucs = []
    make_perm, _ = get_model_fn(model, balanced, iterations=50)
    for pi in range(n_perm):
        prng = np.random.RandomState(rng_seed * 1000 + pi)
        ys = prng.permutation(y)
        pm_m = make_perm()
        pm_m.fit(X[:split], ys[:split])
        try:
            pa = roc_auc_score(ys[split:], pm_m.predict_proba(X[split:])[:, 1])
        except Exception:
            pa = 0.5
        perm_aucs.append(pa)
    perm_passed = auc > np.percentile(perm_aucs, 95)

    return {
        "auc": auc,
        "pbo": pbo,
        "net_sr": net_sr,
        "gross_sr": gross_sr,
        "var_sr_is": var_sr_is,
        "long_pct": long_pct,
        "perm_passed": perm_passed,
        "events": len(data),
        "model": model_name,
    }


GATES = [
    ("AUC > 0.55", lambda d: d["auc"] > 0.55),
    ("Permutation", lambda d: d["perm_passed"].astype(bool)),
    ("PBO < 0.20", lambda d: d["pbo"] < 0.20),
    ("Net SR > 0", lambda d: d["net_sr"] > 0),
    ("Full VALID", lambda d: (d["auc"] > 0.55) & (d["pbo"] < 0.20) & (d["net_sr"] > 0)),
]


def run_setting(setting_key, data_source="synthetic", model="auto",
                n_mc=None, n_perm=None, run_cfg=None):
    run_cfg = run_cfg or config.FULL
    n_mc = n_mc or run_cfg.n_mc
    n_perm = n_perm or run_cfg.n_perm
    cfg = SETTINGS[setting_key]
    print(f"\n{'=' * 70}")
    print(f"MC NULL PIPELINE: {cfg['label']} — {n_mc} iterations ({data_source} data)")
    print(f"{'=' * 70}")

    df = load_ohlcv(data_source, cfg["asset"], n_bars=run_cfg.n_bars)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    ret_actual = df["close"].pct_change().dropna().values
    n_days = len(ret_actual)
    ref_index = df.index

    print(f"  Source: {df.attrs.get('source', data_source)} — {n_days} bars")

    results = []
    t0 = time.time()
    for seed in range(n_mc):
        res = run_mc_iteration(seed, ret_actual, n_days, ref_index,
                               model=model, balanced=True, n_perm=n_perm)
        if res is not None:
            res["mc_iter"] = seed
            res["setting"] = setting_key
            res["data_source"] = data_source
            results.append(res)
        if (seed + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (seed + 1) * (n_mc - seed - 1)
            print(f"  [{seed + 1:3d}/{n_mc}] valid={len(results)} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    df_res = pd.DataFrame(results)
    out_path = config.RESULTS_DIR / f"mc_fpr_{setting_key}.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(df_res)} valid iterations)")

    n = len(df_res)
    if n > 0:
        print(f"\n  === {cfg['label']} FPR Summary (n={n}) ===")
        for name, gate in GATES:
            fpr = gate(df_res).mean()
            lo, hi = wilson_ci(fpr, n)
            print(f"  {name:20s}: {fpr * 100:5.1f}% [{lo * 100:.1f}%, {hi * 100:.1f}%]")
        var_sr = df_res["var_sr_is"].dropna()
        print(f"\n  Mean AUC: {df_res['auc'].mean():.3f} ± {df_res['auc'].std():.3f}")
        if len(var_sr) > 0:
            print(f"  Var(SR_IS): mean={var_sr.mean():.3f}, 95th={var_sr.quantile(0.95):.3f}")

    return df_res


def summarize(frames):
    """Write per-setting x per-gate FPR summary with Wilson CIs."""
    rows = []
    for setting_key, df_res in frames.items():
        n = len(df_res)
        if n == 0:
            continue
        for name, gate in GATES:
            fpr = gate(df_res).mean()
            lo, hi = wilson_ci(fpr, n)
            rows.append({"setting": setting_key, "gate": name, "fpr": fpr,
                         "wilson_lo": lo, "wilson_hi": hi, "n": n})
    out = config.RESULTS_DIR / "mc_fpr_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=list(SETTINGS.keys()))
    parser.add_argument("--all", action="store_true", help="Run all settings")
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--model", choices=["auto", "catboost", "hgb"], default="auto")
    parser.add_argument("--n-mc", type=int, default=None)
    parser.add_argument("--n-perm", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="CI tier (n_mc=10)")
    args = parser.parse_args()

    run_cfg = config.get_run_config(smoke=args.smoke)
    kw = dict(data_source=args.data, model=args.model,
              n_mc=args.n_mc, n_perm=args.n_perm, run_cfg=run_cfg)

    if args.all:
        frames = {key: run_setting(key, **kw) for key in SETTINGS}
        summarize(frames)
    elif args.setting:
        run_setting(args.setting, **kw)
    else:
        parser.print_help()

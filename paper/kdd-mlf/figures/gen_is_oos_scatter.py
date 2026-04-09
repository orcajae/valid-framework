#!/usr/bin/env python3
"""
Figure: IS vs OOS Sharpe Ratio Scatter (Wiecki et al. parallel)
Uses CPCV results: 9 model configs with IS and OOS SR.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Load CPCV data (9 model configs) ─────────────────────────────
df = pd.read_csv("../../../results/cpcv_pbo_results.csv")
is_sr = df["is_sharpe_all_mean"].values
oos_sr = df["cpcv_sharpe_all_mean"].values
degradation = df["degradation_all"].values

# ── Also load walk-forward 32 windows ────────────────────────────
wf = pd.read_csv("../../../results/paper_stats/walk_forward.csv")
# Filter extreme outliers (>10 or <-10 are clearly errors from circuit breaker edge cases)
wf_clean = wf[(wf["oos_sr"].abs() < 10) & (wf["is_sr"].abs() < 10)]
wf_is = wf_clean["is_sr"].values
wf_oos = wf_clean["oos_sr"].values

# ── Correlations ─────────────────────────────────────────────────
rho_cpcv, p_cpcv = stats.spearmanr(is_sr, oos_sr)
rho_wf, p_wf = stats.spearmanr(wf_is, wf_oos)
r2_cpcv = np.corrcoef(is_sr, oos_sr)[0, 1] ** 2
r2_wf = np.corrcoef(wf_is, wf_oos)[0, 1] ** 2

print(f"CPCV: Spearman ρ = {rho_cpcv:.3f} (p={p_cpcv:.3f}), R² = {r2_cpcv:.4f}")
print(f"WF:   Spearman ρ = {rho_wf:.3f} (p={p_wf:.3f}), R² = {r2_wf:.4f}")
print(f"CPCV: n={len(is_sr)}, WF: n={len(wf_is)} (after filtering)")

# ── Plot ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.666, 2.8))

# Panel A: CPCV IS vs OOS
ax1.scatter(is_sr, oos_sr, c="#4878CF", s=40, alpha=0.8, edgecolors="white",
            linewidth=0.5, zorder=5)
lims1 = [min(is_sr.min(), oos_sr.min()) - 0.5,
         max(is_sr.max(), oos_sr.max()) + 0.5]
ax1.plot(lims1, lims1, "k--", alpha=0.3, linewidth=0.8, label="Perfect transfer")
ax1.set_xlabel("In-Sample SR")
ax1.set_ylabel("Out-of-Sample SR")
ax1.set_title(f"CPCV ($\\rho$={rho_cpcv:.2f}, $R^2$={r2_cpcv:.3f})", fontsize=8)
ax1.legend(fontsize=6)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel B: Walk-Forward IS vs OOS
ax2.scatter(wf_is, wf_oos, c="#C44E52", s=40, alpha=0.7, edgecolors="white",
            linewidth=0.5, zorder=5)
lims2 = [min(wf_is.min(), wf_oos.min()) - 0.3,
         max(wf_is.max(), wf_oos.max()) + 0.3]
ax2.plot(lims2, lims2, "k--", alpha=0.3, linewidth=0.8, label="Perfect transfer")
ax2.set_xlabel("In-Sample SR")
ax2.set_ylabel("Out-of-Sample SR")
ax2.set_title(f"Walk-Forward ($\\rho$={rho_wf:.2f}, $R^2$={r2_wf:.3f})", fontsize=8)
ax2.legend(fontsize=6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()

OUT = Path(__file__).resolve().parent
fig.savefig(OUT / "fig_is_oos_scatter.pdf")
fig.savefig(OUT / "fig_is_oos_scatter.png")
print(f"✓ Saved to {OUT / 'fig_is_oos_scatter.pdf'}")

# ── Save JSON ────────────────────────────────────────────────────
import json
result = {
    "cpcv": {
        "n_configs": int(len(is_sr)),
        "spearman_rho": round(rho_cpcv, 3),
        "spearman_p": round(p_cpcv, 3),
        "r_squared": round(r2_cpcv, 4),
    },
    "walk_forward": {
        "n_windows": int(len(wf_is)),
        "n_total_windows": int(len(wf)),
        "n_filtered": int(len(wf) - len(wf_clean)),
        "spearman_rho": round(rho_wf, 3),
        "spearman_p": round(p_wf, 3),
        "r_squared": round(r2_wf, 4),
    },
    "known_benchmark": "Wiecki et al. (2016): R² < 0.025 on 888 strategies",
}
with open(OUT.parent / "results" / "is_oos_correlation.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"✓ JSON saved")

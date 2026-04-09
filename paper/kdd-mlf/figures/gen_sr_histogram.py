#!/usr/bin/env python3
"""
Figure 1: Distribution of Net Sharpe Ratios across 340 strategy variants.
For KDD-MLF 2026 — VALID paper.

ACM sigconf single-column width = 3.333 inches.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Data ─────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parent.parent.parent.parent
VARIANTS = DATA / "valid-framework/results/reference/all_482_variants.csv"
OUT = Path(__file__).resolve().parent / "fig1_sr_distribution.pdf"
OUT_PNG = Path(__file__).resolve().parent / "fig1_sr_distribution.png"

df = pd.read_csv(VARIANTS)
df = df[~df["strategy_id"].str.startswith("MC_")].copy()
srs = df["net_sr_18bp"].astype(float).values

BENCH_SR = 0.917
BREAKEVEN = 0.0

# ── Plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.333, 2.2))

# Histogram
bins = np.arange(-3.5, 2.5, 0.15)
n, bins_out, patches = ax.hist(srs, bins=bins, color="#4878CF", alpha=0.75,
                                edgecolor="white", linewidth=0.3)

# Color negative bars darker
for patch, b in zip(patches, bins_out[:-1]):
    if b + 0.075 < BREAKEVEN:
        patch.set_facecolor("#C44E52")
        patch.set_alpha(0.65)

# Reference lines
ax.axvline(BENCH_SR, color="#2CA02C", linewidth=1.5, linestyle="-",
           label=f"DM Benchmark (SR={BENCH_SR})", zorder=5)
ax.axvline(BREAKEVEN, color="#D62728", linewidth=1.2, linestyle="--",
           label="Break-even (SR=0)", zorder=5)

# Annotations
neg_pct = 100 * (srs < 0).sum() / len(srs)
beat_pct = 100 * (srs > BENCH_SR).sum() / len(srs)

ax.annotate(f"{neg_pct:.0f}% negative SR",
            xy=(-1.0, ax.get_ylim()[1] * 0.92),
            fontsize=7.5, color="#C44E52", fontweight="bold",
            ha="center")
ax.annotate(f"{beat_pct:.1f}% beat\nbenchmark",
            xy=(BENCH_SR + 0.35, ax.get_ylim()[1] * 0.72),
            fontsize=7, color="#2CA02C", fontweight="bold",
            ha="center")

ax.set_xlabel("Net Sharpe Ratio (18 bp round-trip)")
ax.set_ylabel("Count")
ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
ax.set_xlim(-3.5, 2.5)

# Clean up
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT)
fig.savefig(OUT_PNG)
print(f"✓ Saved {OUT}")
print(f"✓ Saved {OUT_PNG}")
print(f"  N={len(srs)}, negative={neg_pct:.1f}%, beat benchmark={beat_pct:.1f}%")

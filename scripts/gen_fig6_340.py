#!/usr/bin/env python3
"""
Extended-version distribution figure, regenerated from the 340-variant corpus.

Replaces `fig6_482_distribution.png`, whose benchmark line was drawn at the
maximum of the DM/baseline rows (0.954, the benchmark evaluated at 0 bp) rather
than at the 0.917 benchmark the paper compares against, and whose filename
carried a deprecated variant count.

  source : results/reference/variants_340.csv
  output : $FIG6_OUT (default <repo>/paper/latex/figures/fig6_sr_distribution.pdf)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "reference" / "variants_340.csv"
OUT = Path(os.environ.get("FIG6_OUT",
                          ROOT / "paper/latex/figures/fig6_sr_distribution.pdf"))

BENCH_SR = 0.917
NEG_COLOR, NEG_EDGE = "#B4B2A9", "#5F5E5A"
POS_COLOR, POS_EDGE = "#B5D4F4", "#185FA5"
BENCH_COLOR = "#185FA5"

df = pd.read_csv(SRC)
df = df[~df["strategy_id"].str.startswith("MC_")].copy()
srs = df["net_sr_18bp"].astype(float).values
assert len(srs) == 340, f"expected 340 variants, got {len(srs)}"

neg_pct = 100 * (srs < 0).sum() / len(srs)
beat_pct = 100 * (srs > BENCH_SR).sum() / len(srs)

fig, ax = plt.subplots(figsize=(4.6, 2.9))
bins = np.arange(-3.5, 2.5, 0.15)
n, bins_out, patches = ax.hist(srs, bins=bins, color=POS_COLOR, alpha=0.85,
                               edgecolor=POS_EDGE, linewidth=0.3)
for patch, b in zip(patches, bins_out[:-1]):
    if b + 0.075 < 0.0:
        patch.set_facecolor(NEG_COLOR)
        patch.set_edgecolor(NEG_EDGE)
        patch.set_alpha(0.75)

ax.axvline(BENCH_SR, color=BENCH_COLOR, linewidth=1.3, zorder=5)
ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle="--", zorder=5)

ymax = ax.get_ylim()[1]
ax.annotate(f"{neg_pct:.0f}% negative", xy=(-2.5, ymax * 0.62), fontsize=8.5,
            fontweight="bold", color=NEG_EDGE, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#D3D1C7",
                      edgecolor=NEG_EDGE, linewidth=0.4, alpha=0.9))
ax.annotate(f"{beat_pct:.1f}% above\nbenchmark", xy=(1.15, 3),
            xytext=(1.85, ymax * 0.52), fontsize=7.5, fontweight="bold",
            color="#FFFFFF", ha="center",
            arrowprops=dict(arrowstyle="->,head_width=0.15",
                            color=BENCH_COLOR, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#378ADD",
                      edgecolor=BENCH_COLOR, linewidth=0.4, alpha=0.9))

ax.set_xlabel("Net Sharpe Ratio (18 bp round-trip)")
ax.set_ylabel("Number of strategy variants")
ax.set_xlim(-3.5, 2.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.legend(handles=[
    Patch(facecolor=NEG_COLOR, alpha=0.75, edgecolor=NEG_EDGE, label="Negative SR"),
    Patch(facecolor=POS_COLOR, alpha=0.85, edgecolor=POS_EDGE, label="Positive SR"),
    plt.Line2D([0], [0], color=BENCH_COLOR, linewidth=1.3,
               label=f"DM Benchmark (SR={BENCH_SR})"),
    plt.Line2D([0], [0], color="#333333", linewidth=0.8, linestyle="--",
               label="Break-even (SR=0)"),
], fontsize=6.5, loc="upper left", framealpha=0.9, edgecolor="gray",
    fancybox=False)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"))
print(f"Saved {OUT}")
print(f"  N={len(srs)}, negative={neg_pct:.1f}%, above benchmark={beat_pct:.1f}%, "
      f"benchmark line at {BENCH_SR}")

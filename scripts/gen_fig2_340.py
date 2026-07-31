#!/usr/bin/env python3
"""
Figure 2: distribution of net Sharpe ratios across the 340-variant corpus.

Replaces paper/kdd-mlf/figures/gen_sr_histogram.py, which read the legacy
`all_482_variants.csv` (540 rows) and filtered its 200 Monte-Carlo null rows to
arrive at the same 340 values. This version reads the corpus file directly and
embeds TrueType fonts (pdf.fonttype 42) instead of Type 3.

Style is unchanged: same figure size, bins, palette, annotations and legend.
No experiment is run — the script only re-plots stored values.

  source : results/reference/variants_340.csv
  output : $FIG2_OUT (default ~/jwquant/paper/kdd-mlf/figures/fig2_sr_distribution.pdf)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42     # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
})

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ROOT / "results" / "reference" / "variants_340.csv"
OUT = Path(os.environ.get(
    "FIG2_OUT",
    Path.home() / "jwquant/paper/kdd-mlf/figures/fig2_sr_distribution.pdf"))
OUT_PNG = OUT.with_suffix(".png")

df = pd.read_csv(VARIANTS)
df = df[~df["strategy_id"].str.startswith("MC_")].copy()   # no-op guard
srs = df["net_sr_18bp"].astype(float).values
assert len(srs) == 340, f"expected 340 variants, got {len(srs)}"

BENCH_SR = 0.917
BREAKEVEN = 0.0
neg_pct = 100 * (srs < 0).sum() / len(srs)
beat_pct = 100 * (srs > BENCH_SR).sum() / len(srs)

# ── Colors (monochrome blue + gray) ──
NEG_COLOR   = "#B4B2A9"
NEG_EDGE    = "#5F5E5A"
POS_COLOR   = "#B5D4F4"
POS_EDGE    = "#185FA5"
BENCH_COLOR = "#185FA5"
ANNO_NEG_BG = "#D3D1C7"
ANNO_NEG_TX = "#5F5E5A"
ANNO_POS_BG = "#378ADD"
ANNO_POS_TX = "#FFFFFF"

fig, ax = plt.subplots(figsize=(3.333, 2.2))

bins = np.arange(-3.5, 2.5, 0.15)
n, bins_out, patches = ax.hist(srs, bins=bins, color=POS_COLOR, alpha=0.85,
                               edgecolor=POS_EDGE, linewidth=0.3)

for patch, b in zip(patches, bins_out[:-1]):
    if b + 0.075 < BREAKEVEN:
        patch.set_facecolor(NEG_COLOR)
        patch.set_edgecolor(NEG_EDGE)
        patch.set_alpha(0.75)

ax.axvline(BENCH_SR, color=BENCH_COLOR, linewidth=1.3, linestyle="-", zorder=5)
ax.axvline(BREAKEVEN, color="#333333", linewidth=0.8, linestyle="--", zorder=5)

ymax = ax.get_ylim()[1]

ax.annotate(f"{neg_pct:.0f}% negative",
            xy=(-2.5, ymax * 0.55), fontsize=7.5, fontweight="bold",
            color=ANNO_NEG_TX, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=ANNO_NEG_BG,
                      edgecolor=NEG_EDGE, linewidth=0.4, alpha=0.9))

ax.annotate(f"{beat_pct:.1f}% beat\nbenchmark",
            xy=(1.15, 3), xytext=(1.8, ymax * 0.50),
            fontsize=6.5, fontweight="bold", color=ANNO_POS_TX, ha="center",
            arrowprops=dict(arrowstyle="->,head_width=0.15",
                            color=BENCH_COLOR, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", facecolor=ANNO_POS_BG,
                      edgecolor=BENCH_COLOR, linewidth=0.4, alpha=0.9))

ax.set_xlabel("Net Sharpe Ratio (18 bp round-trip)")
ax.set_ylabel("Number of strategy variants")

legend_elements = [
    Patch(facecolor=NEG_COLOR, alpha=0.75, edgecolor=NEG_EDGE, label="Negative SR"),
    Patch(facecolor=POS_COLOR, alpha=0.85, edgecolor=POS_EDGE, label="Positive SR"),
    plt.Line2D([0], [0], color=BENCH_COLOR, linewidth=1.3, linestyle="-",
               label=f"DM Benchmark (SR={BENCH_SR})"),
    plt.Line2D([0], [0], color="#333333", linewidth=0.8, linestyle="--",
               label="Break-even (SR=0)"),
]
ax.legend(handles=legend_elements, fontsize=5.5, loc="upper left",
          framealpha=0.9, edgecolor="gray", fancybox=False)

ax.set_xlim(-3.5, 2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT_PNG)
print(f"Saved {OUT}")
print(f"Saved {OUT_PNG}")
print(f"  N={len(srs)}, negative={neg_pct:.1f}%, beat={beat_pct:.1f}%")
print(f"  bins={len(bins) - 1}, max bin height={int(n.max())}")

#!/usr/bin/env python3
"""
Figure 4 (the PBO paradox), regenerated from the corpus and the Monte Carlo run.

Replaces `fig2_pbo_paradox.png`, a 2026-04-10 raster in default matplotlib
style (sans-serif, Type 3 fonts, saturated red/green/orange) that had no
generator in the repository.

Panel (a) was a scatter of AUC against PBO. Its y axis carries no variance:
every one of the 209 corpus variants for which PBO was computed has PBO
exactly 1.000, so the "cloud" was a single horizontal line of points and the
caption's "nearly all configurations receive PBO close to 1.0" understated the
result. The panel now says what the data says -- every variant sits at the
ceiling -- and draws the V4 acceptance threshold so the distance to it is the
visible quantity rather than empty space.

Panel (b) keeps the null-distribution histogram and gains the absolute
flatness threshold, the second of the two criteria the text reports as
disagreeing.

  sources : results/reference/variants_340.csv          (auc, pbo)
            results/reference/monte_carlo_fpr_200.csv   (var_sr_is null, n=200)
            $VAR_SR_IS_CSV                              (real Var(SR_IS))
  output  : $FIG_PBO_OUT (default <repo>/paper/latex/figures/fig2_pbo_paradox.pdf)

The real-data Var(SR_IS) lives in `results/paper_stats/var_sr_is.csv`, which is
an experiment output that is not tracked in this repository; point at it with
VAR_SR_IS_CSV or the script falls back to the author's local layout, as
`gen_fig5_regime_real.py` does for the regime artifact.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "results" / "reference" / "variants_340.csv"
MC = ROOT / "results" / "reference" / "monte_carlo_fpr_200.csv"
REAL = Path(os.environ.get(
    "VAR_SR_IS_CSV", Path.home() / "jwquant/results/paper_stats/var_sr_is.csv"))
OUT = Path(os.environ.get("FIG_PBO_OUT",
                          ROOT / "paper/latex/figures/fig2_pbo_paradox.pdf"))

PBO_THRESHOLD = 0.20        # V4 acceptance threshold
AUC_CRITERION = 0.55        # the AUC-only criterion the ablation tests
FLAT_ABSOLUTE = 0.01        # absolute flatness threshold


def main():
    corpus = pd.read_csv(CORPUS)
    corpus = corpus[~corpus["strategy_id"].str.startswith("MC_")]
    pts = corpus.dropna(subset=["auc", "pbo"])
    auc, pbo = pts["auc"].to_numpy(float), pts["pbo"].to_numpy(float)

    null = pd.read_csv(MC)["var_sr_is"].dropna().to_numpy(float)
    p95 = float(np.percentile(null, 95))

    if not REAL.exists():
        raise SystemExit(
            f"real Var(SR_IS) artifact not found at {REAL}\n"
            f"set VAR_SR_IS_CSV=/path/to/var_sr_is.csv")
    real = float(pd.read_csv(REAL)["mean_var_sr_is_per_fold"].iloc[0])

    fs.apply(base=9.0)
    fig, axes = plt.subplots(1, 2, figsize=(fs.width(1.0), 2.85))

    # --- (a) where the corpus lands relative to the two criteria ---------
    ax = axes[0]
    ax.scatter(auc, pbo, s=13, facecolor=fs.FILL_MID, edgecolor=fs.STROKE,
               linewidth=0.4, alpha=0.85, zorder=4)
    ax.axhline(PBO_THRESHOLD, color=fs.ACCENT, ls=(0, (4, 3)), lw=0.9, zorder=3)
    ax.axvline(AUC_CRITERION, color=fs.GRAY_STROKE, ls=(0, (1, 2)), lw=0.9,
               zorder=3)
    ax.text(0.415, PBO_THRESHOLD + 0.035, f"V4 accepts below {PBO_THRESHOLD:.2f}",
            fontsize=7.5, color=fs.ACCENT, ha="left", va="bottom")
    ax.text(AUC_CRITERION + 0.006, 0.30, f"AUC {AUC_CRITERION}", fontsize=7.5,
            color=fs.GRAY_STROKE, ha="left", va="bottom", rotation=90)
    # Two lines, so the label clears the y axis on the left and the AUC
    # criterion line on the right.
    ax.annotate(f"all {len(pts)} variants\nat PBO = 1.000",
                xy=(0.475, 0.985), xytext=(0.475, 0.86),
                fontsize=7.5, color=fs.INK_DEEP, ha="center", va="top",
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.12",
                                color=fs.INK, lw=0.7))
    ax.set_xlim(0.41, 0.66)
    ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("AUC (balanced)")
    ax.set_ylabel("PBO")
    fs.panel(ax, "(a) PBO against the V4 threshold")
    fs.despine(ax)
    fs.hgrid(ax)

    # --- (b) parameter-space variance against the null -------------------
    ax = axes[1]
    ax.hist(null, bins=np.linspace(0, 0.60, 25), color=fs.FILL_MID,
            edgecolor=fs.STROKE, linewidth=0.35, zorder=3)
    # Drawn to a ceiling rather than the full axis height: at full height the
    # null percentile line at 0.328 runs up through the legend block.
    TOP = 24
    ax.vlines(real, 0, TOP, color=fs.INK_DEEP, lw=1.4, zorder=5)
    ax.vlines(p95, 0, TOP, color=fs.ACCENT, ls=(0, (4, 3)), lw=1.0, zorder=5)
    ax.vlines(FLAT_ABSOLUTE, 0, TOP, color=fs.GRAY_STROKE, ls=(0, (1, 2)),
              lw=0.9, zorder=4)
    ax.set_xlim(0, 0.60)
    ax.set_ylim(0, 34)          # headroom: the legend sat on the tallest bin
    ax.set_xlabel(r"$\mathrm{Var}(\mathrm{SR}_{\mathrm{IS}})$")
    ax.set_ylabel("Monte Carlo iterations")
    fs.panel(ax, "(b) Real value against the null")
    fs.despine(ax)
    fs.hgrid(ax)
    ax.legend(handles=[
        Line2D([0], [0], color=fs.INK_DEEP, lw=1.4,
               label=f"Real data: {real:.3f}"),
        Line2D([0], [0], color=fs.GRAY_STROKE, ls=(0, (1, 2)), lw=0.9,
               label=f"Absolute flatness: {FLAT_ABSOLUTE:.2f}"),
        Line2D([0], [0], color=fs.ACCENT, ls=(0, (4, 3)), lw=1.0,
               label=f"Null 95th pct: {p95:.3f}"),
    ], fontsize=7.5, loc="upper right", frameon=False, handlelength=1.7,
        borderpad=0.2, labelspacing=0.35)

    fig.tight_layout()
    fs.save(fig, OUT)
    print(f"Saved {OUT}")
    print(f"  (a) n={len(pts)} variants with both AUC and PBO; "
          f"PBO range [{pbo.min():.3f}, {pbo.max():.3f}]; "
          f"AUC range [{auc.min():.3f}, {auc.max():.3f}]")
    print(f"  (b) null n={len(null)}, 95th pct={p95:.4f}, real={real:.4f}, "
          f"absolute threshold={FLAT_ABSOLUTE}")


if __name__ == "__main__":
    main()

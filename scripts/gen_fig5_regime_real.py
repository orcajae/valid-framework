#!/usr/bin/env python3
"""
Figure 5 (regime-conditional performance), regenerated from the stored
regime artifact.

The v3.1 figure carried a third series, "Best ML (1h balanced)", whose bars
were hard-coded: `ml_srs = [0.30, 0.50]  # approximate — ML was poor in both
regimes` (src/paper_figures.py L292). No regime-conditional Sharpe ratio was
ever computed for any model variant, so that series is dropped rather than
re-estimated. The two remaining series are read from the artifact.

  source : ~/jwquant/paper/kdd-mlf/results/regime_conditional.json
           (cross-check: ~/jwquant/results/paper_stats/regime_performance.csv)
  output : $FIG5_OUT (default ~/jwquant/valid-framework/paper/latex/figures/
                      fig5_regime_real.pdf)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

fs.apply(base=9.0)

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get(
    "REGIME_JSON", Path.home() / "jwquant/paper/kdd-mlf/results/regime_conditional.json"))
OUT = Path(os.environ.get("FIG5_OUT",
                          ROOT / "paper/latex/figures/fig5_regime_real.pdf"))

BLUE, EDGE = fs.FILL_MID, fs.STROKE
GRAY, GRAY_EDGE = fs.GRAY_FILL, fs.GRAY_STROKE

# Included at 0.78\columnwidth, so author at that width: the figure was
# drawn 3.4 in wide and scaled up by 1.25, which is why its labels read a
# quarter larger than every other figure in the paper.
WIDTH = fs.width(0.78)


def main():
    d = json.load(open(SRC))
    calm, stressed = d["calm"], d["stressed"]
    labels = [f"Calm ({calm['pct']:.0f}% of days)",
              f"Stressed ({stressed['pct']:.0f}% of days)"]
    dm = [calm["dm_sr"], stressed["dm_sr"]]
    bnh = [calm["bnh_sr"], stressed["bnh_sr"]]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, dm, w, color=BLUE, edgecolor=EDGE, linewidth=0.5,
           label="DM benchmark")
    ax.bar(x + w / 2, bnh, w, color=GRAY, edgecolor=GRAY_EDGE, linewidth=0.5,
           label="Buy-and-hold")
    for i, (a, b) in enumerate(zip(dm, bnh)):
        ax.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", fontsize=8,
                color=fs.INK_DEEP)
        ax.text(i + w / 2, b + 0.03, f"{b:.2f}", ha="center", fontsize=8,
                color=fs.INK_DEEP)
        ax.annotate(f"$+${a - b:.2f}", xy=(i, max(a, b) + 0.20), ha="center",
                    fontsize=8, color=EDGE)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_ylim(0, 1.75)
    ax.legend(fontsize=8, frameon=False, loc="upper left", handlelength=1.6)
    fs.despine(ax)
    fs.hgrid(ax)

    plt.tight_layout()
    fs.save(fig, OUT)
    print(f"Saved {OUT}")
    print(f"  regime method : {d['regime_method']}")
    print(f"  calm     : {calm['pct']}% ({calm['days']}d)  DM {calm['dm_sr']:.4f}  BnH {calm['bnh_sr']:.4f}")
    print(f"  stressed : {stressed['pct']}% ({stressed['days']}d)  DM {stressed['dm_sr']:.4f}  BnH {stressed['bnh_sr']:.4f}")
    print(f"  spread   : calm +{calm['dm_sr'] - calm['bnh_sr']:.4f}  stressed +{stressed['dm_sr'] - stressed['bnh_sr']:.4f}")


if __name__ == "__main__":
    main()

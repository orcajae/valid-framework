#!/usr/bin/env python3
"""
Figure 3 (bull bias), regenerated from the 340-variant corpus.

Replaces the v3.1 figure, whose four panels used hard-coded prediction shares
(91/9, 60.4/39.6, 93/7) that do not match the paper's own bull-bias table, and
whose bear-market panel plotted equity curves built from `np.random.choice`
signals (src/paper_figures.py L69-98).

The v3.2 layout also fixes two collisions. In panel (a) the legend sat on the
top of the first bar; in panel (b) a "0.50 (chance)" caption ran into the first
bar and the reference line. Both labels are now redundant with the y axis, so
the two panels share one legend beneath the figure and the reference line is
keyed there rather than annotated in the plot area.

Every value here is read from results/reference/variants_340.csv, the same
source as the bull-bias table. No model is trained and no backtest is run.

  source : results/reference/variants_340.csv
  output : $FIG1_OUT (default <repo>/paper/latex/figures/fig1_bullbias_corpus.pdf)
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "reference" / "variants_340.csv"
OUT = Path(os.environ.get("FIG1_OUT",
                          ROOT / "paper/latex/figures/fig1_bullbias_corpus.pdf"))

# (label, unbalanced row, balanced row) — balanced AUC taken from the balanced row
GROUPS = [("BTC\nCatBoost", "EXT_BTC_1h_CatBoost", "EXT_BTC_1h_CatBoost"),
          ("ETH\nCatBoost", "EXT_ETH_1h_CatBoost", "EXT_ETH_1h_CatBoost"),
          ("SOL\nCatBoost", "EXT_SOL_1h_CatBoost", "EXT_SOL_1h_CatBoost"),
          ("BTC\nLSTM", "BB_LSTM_unbalanced", "BB_LSTM_balanced")]


def main():
    rows = {r["strategy_id"]: r for r in csv.DictReader(open(SRC))}
    labels, unbal, bal, auc = [], [], [], []
    for lab, u_id, b_id in GROUPS:
        u, b = rows[u_id], rows[b_id]
        labels.append(lab)
        unbal.append(100 * float(u["unbal_long_pct"]))
        # dedicated balanced rows (BB_*_balanced) record their long share in
        # the unbalanced column; combined rows use bal_long_pct
        bal_raw = b["bal_long_pct"] or b["unbal_long_pct"]
        bal.append(100 * float(bal_raw))
        auc.append(float(b["auc"]))

    fs.apply(base=9.0)
    fig, axes = plt.subplots(1, 2, figsize=(fs.width(1.0), 2.85))
    x = np.arange(len(labels))
    w = 0.38

    # --- (a) directional prediction share --------------------------------
    ax = axes[0]
    ax.bar(x - w / 2, unbal, w, color=fs.GRAY_FILL, edgecolor=fs.GRAY_STROKE,
           linewidth=0.5, zorder=3)
    ax.bar(x + w / 2, bal, w, color=fs.FILL_MID, edgecolor=fs.STROKE,
           linewidth=0.5, zorder=3)
    # 50 is already a y tick, so the line needs no in-plot caption; it is
    # keyed in the shared legend instead.
    ax.axhline(50, color=fs.ACCENT, ls=(0, (4, 3)), lw=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Predicted "long" (%)')
    fs.panel(ax, "(a) Directional prediction share")
    fs.despine(ax)
    fs.hgrid(ax)

    # --- (b) discriminative power after balancing ------------------------
    ax = axes[1]
    ax.bar(x, auc, 0.5, color=fs.FILL_MID, edgecolor=fs.STROKE, linewidth=0.5,
           zorder=3)
    ax.axhline(0.50, color=fs.ACCENT, ls=(0, (4, 3)), lw=0.9, zorder=4)
    # Every label sits just inside the top of its bar. Placing them above the
    # bars instead would put the 0.493 one on the no-skill line, and splitting
    # the rule by sign left the four labels inconsistently aligned.
    for i, v in enumerate(auc):
        ax.text(i, v - 0.0028, f"{v:.3f}", ha="center", va="top", fontsize=7.5,
                color=fs.INK_DEEP, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.44, 0.545)
    ax.set_yticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54])
    ax.set_ylabel("AUC after balancing")
    fs.panel(ax, "(b) Discriminative power after balancing")
    fs.despine(ax)
    fs.hgrid(ax)

    # --- one legend for both panels, below the figure --------------------
    fig.legend(handles=[
        Patch(facecolor=fs.GRAY_FILL, edgecolor=fs.GRAY_STROKE, linewidth=0.5,
              label="Unbalanced"),
        Patch(facecolor=fs.FILL_MID, edgecolor=fs.STROKE, linewidth=0.5,
              label="Balanced"),
        Line2D([0], [0], color=fs.ACCENT, ls=(0, (4, 3)), lw=0.9,
               label="No-skill reference (50% share, 0.50 AUC)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False,
        handlelength=1.6, columnspacing=2.0)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fs.save(fig, OUT)
    print(f"Saved {OUT}")
    for lab, u, b, a in zip(labels, unbal, bal, auc):
        print(f"  {lab.replace(chr(10), ' '):<16} unbal={u:5.1f}%  "
              f"bal={b:5.1f}%  AUC={a:.3f}")


if __name__ == "__main__":
    main()

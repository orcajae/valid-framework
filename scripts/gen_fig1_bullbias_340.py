#!/usr/bin/env python3
"""
Figure 1 (bull bias), regenerated from the 340-variant corpus.

Replaces the v3.1 figure, whose four panels used hard-coded prediction shares
(91/9, 60.4/39.6, 93/7) that do not match the paper's own bull-bias table, and
whose bear-market panel plotted equity curves built from `np.random.choice`
signals (src/paper_figures.py L69-98).

Every value here is read from results/reference/variants_340.csv, the same
source as the bull-bias table. No model is trained and no backtest is run.

  output : $FIG1_OUT (default ~/jwquant/valid-framework/paper/latex/figures/
                      fig1_bullbias_corpus.pdf)
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

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
OUT = Path(os.environ.get("FIG1_OUT",
                          ROOT / "paper/latex/figures/fig1_bullbias_corpus.pdf"))

# (label, unbalanced row, balanced row) — balanced AUC taken from the balanced row
GROUPS = [("BTC\nCatBoost", "EXT_BTC_1h_CatBoost", "EXT_BTC_1h_CatBoost"),
          ("ETH\nCatBoost", "EXT_ETH_1h_CatBoost", "EXT_ETH_1h_CatBoost"),
          ("SOL\nCatBoost", "EXT_SOL_1h_CatBoost", "EXT_SOL_1h_CatBoost"),
          ("BTC\nLSTM", "BB_LSTM_unbalanced", "BB_LSTM_balanced")]

GRAY, BLUE, EDGE = "#B4B2A9", "#B5D4F4", "#185FA5"
GRAY_EDGE = "#5F5E5A"


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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(labels))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, unbal, w, color=GRAY, edgecolor=GRAY_EDGE, linewidth=0.5,
           label="Unbalanced")
    ax.bar(x + w / 2, bal, w, color=BLUE, edgecolor=EDGE, linewidth=0.5,
           label="Balanced")
    ax.axhline(50, color="#333333", ls=(0, (4, 3)), lw=0.9)
    ax.text(-0.48, 52, "50%", fontsize=8, color="#333333", ha="left")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Predicted "long" (%)')
    ax.set_title("(a) Directional prediction share")
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.bar(x, auc, 0.5, color=BLUE, edgecolor=EDGE, linewidth=0.5)
    ax.axhline(0.50, color="#333333", ls=(0, (4, 3)), lw=0.9)
    ax.text(-0.48, 0.4975, "0.50 (chance)", fontsize=8,
            color="#333333", ha="left", va="top")
    for i, v in enumerate(auc):
        off, va = (0.005, "bottom") if v >= 0.50 else (-0.005, "top")
        ax.text(i, v + off, f"{v:.3f}", ha="center", va=va, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.44, 0.56)
    ax.set_ylabel("AUC after balancing")
    ax.set_title("(b) Discriminative power after balancing")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"))
    print(f"Saved {OUT}")
    for lab, u, b, a in zip(labels, unbal, bal, auc):
        print(f"  {lab.replace(chr(10), ' '):<16} unbal={u:5.1f}%  bal={b:5.1f}%  AUC={a:.3f}")


if __name__ == "__main__":
    main()

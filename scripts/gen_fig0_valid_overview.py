#!/usr/bin/env python3
"""
Figure 2: the VALID framework overview.

Replaces `fig0_valid_overview.png`, a 2026-04-10 raster with no generator in
the repository. Its defects were all in the drawing, not the content: adjacent
gate boxes were laid out with a gap smaller than their own rounded-corner
padding, so every internal border doubled up; the "Key Insight" band overlapped
both the Stage 1 container and the Bull Bias box beneath it; curved connectors
ran from the gates across the containers to the failure-mode boxes; and the
title sat outside the canvas and was clipped.

The connectors are gone rather than redrawn. Each failure-mode box already
names the items that detect it, so the curves carried no information the text
did not.

The stage split follows the manuscript (Section 4.3): Stage 1 (Reporting) is
V1-V6, V9, V12 and Stage 2 (Deployment) is V7, V8, V10, V11. This differs from
the camera-ready workflow figure, which splits V1-V6 against V7-V12 under the
names Statistical and Economic.

Style follows `paper/kdd-mlf/figures/gen_workflow.py` by way of `figstyle`.
This figure plots no data; it is a diagram of the framework.

  output : $FIG0_OUT (default <repo>/paper/latex/figures/fig0_valid_overview.pdf)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("FIG0_OUT",
                          ROOT / "paper/latex/figures/fig0_valid_overview.pdf"))

# Labels are kept short enough to clear the item code on the same row; the
# full item names are in the VALID items table.
STAGE1 = [("V1-V2", "Class balance"),
          ("V3", "Temporal split"),
          ("V4", "CPCV with PBO"),
          ("V5", "Var(SR$_{\\mathrm{IS}}$)"),
          ("V6", "Permutation tests"),
          ("V9", "Baseline comparison"),
          ("V12", "Code released")]

STAGE2 = [("V7-V8", "Net SR after costs"),
          ("V10", "Bear market eval"),
          ("V11", "Trade frequency")]

MODES = [("Bull bias", "Detected by V1, V2"),
         ("Statistical-economic disconnect", "Detected by V4, V5, V6"),
         ("Cost illusion", "Detected by V7, V8, V11")]

# --- geometry, in inches; every gap is explicit so no two borders touch ---
W, H = fs.TEXT_W, 4.05
COL_W = [0.68, 1.54, 1.38, 0.68]          # strategy, stage 1, stage 2, verdict
# Wide enough for the "all pass" label to sit between two containers.
COL_GAP = 0.375
GATE_H, GATE_GAP = 0.245, 0.09
PAD = 0.10                                 # container padding around its gates
TITLE_H = 0.40                             # container header
MODE_H = 0.62
BAND_GAP = 0.30

col_x = []
x = 0.01
for w in COL_W:
    col_x.append(x)
    x += w + COL_GAP

STAGE1_H = TITLE_H + len(STAGE1) * GATE_H + (len(STAGE1) - 1) * GATE_GAP + 2 * PAD
BOTTOM = MODE_H + BAND_GAP                 # y of the pipeline band's base


def rbox(ax, x, y, w, h, fc, ec, lw=0.9, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=z, clip_on=False))


def arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=6,
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.09",
                                color=fs.STROKE, lw=1.0))


def stage(ax, xi, items, title, subtitle, fill):
    """Draw a stage container with its gate stack, vertically centred."""
    x, w = col_x[xi], COL_W[xi]
    h = TITLE_H + len(items) * GATE_H + (len(items) - 1) * GATE_GAP + 2 * PAD
    y = BOTTOM + (STAGE1_H - h) / 2
    rbox(ax, x, y, w, h, fill, fs.STROKE, lw=1.0, ls=(0, (4, 2)), z=1)
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=fs.INK_DEEP, zorder=3)
    ax.text(x + w / 2, y + h - 0.31, subtitle, ha="center", va="center",
            fontsize=6.6, style="italic", color=fs.INK, zorder=3)

    gy = y + h - TITLE_H - PAD - GATE_H
    for code, label in items:
        rbox(ax, x + PAD, gy, w - 2 * PAD, GATE_H, "#FFFFFF", fs.STROKE,
             lw=0.7, z=2)
        ax.text(x + PAD + 0.07, gy + GATE_H / 2, code, ha="left", va="center",
                fontsize=6.8, fontweight="bold", color=fs.INK_DEEP, zorder=3)
        ax.text(x + w - PAD - 0.07, gy + GATE_H / 2, label, ha="right",
                va="center", fontsize=6.4, color=fs.INK_DEEP, zorder=3)
        gy -= GATE_H + GATE_GAP
    return y + h / 2


def main():
    fs.apply(base=9.0)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    mid = BOTTOM + STAGE1_H / 2

    # --- entry and verdict ------------------------------------------------
    end_h = 0.62
    rbox(ax, col_x[0], mid - end_h / 2, COL_W[0], end_h, "#E8E8E8", "#999999",
         lw=0.9, z=2)
    ax.text(col_x[0] + COL_W[0] / 2, mid, "ML trading\nstrategy", ha="center",
            va="center", fontsize=7.6, fontweight="bold", color="#555555",
            zorder=3, linespacing=1.5)

    rbox(ax, col_x[3], mid - end_h / 2, COL_W[3], end_h, fs.FILL_STRONG,
         fs.INK, lw=1.0, z=2)
    ax.text(col_x[3] + COL_W[3] / 2, mid, "VALID-\ncompliant", ha="center",
            va="center", fontsize=7.6, fontweight="bold", color="#FFFFFF",
            zorder=3, linespacing=1.5)

    # --- the two stages ---------------------------------------------------
    stage(ax, 1, STAGE1, "Stage 1: Reporting", "scientific validity",
          fs.FILL_LIGHT)
    stage(ax, 2, STAGE2, "Stage 2: Deployment", "economic viability",
          fs.FILL_MID)

    for i in range(3):
        x1 = col_x[i] + COL_W[i] + 0.045
        x2 = col_x[i + 1] - 0.045
        arrow(ax, x1, x2, mid)
        ax.text((x1 + x2) / 2, mid + 0.075, "all pass", ha="center",
                va="bottom", fontsize=6.2, style="italic", color=fs.INK)

    # --- failure modes ----------------------------------------------------
    ax.text(0.01, MODE_H + BAND_GAP / 2 - 0.03,
            "Failure modes and the items that detect them",
            ha="left", va="center", fontsize=6.8, style="italic",
            color=fs.INK)

    mw = (W - 0.02 - 2 * COL_GAP) / 3
    for i, (name, detected) in enumerate(MODES):
        mx = 0.01 + i * (mw + COL_GAP)
        rbox(ax, mx, 0.02, mw, MODE_H, fs.FILL_LIGHT, fs.STROKE, lw=0.8, z=2)
        ax.text(mx + mw / 2, 0.02 + MODE_H * 0.63, name, ha="center",
                va="center", fontsize=7.4, fontweight="bold",
                color=fs.INK_DEEP, zorder=3)
        ax.text(mx + mw / 2, 0.02 + MODE_H * 0.26, detected, ha="center",
                va="center", fontsize=6.6, color=fs.INK, zorder=3)

    fs.save(fig, OUT)
    print(f"Saved {OUT}")
    print(f"  Stage 1: {len(STAGE1)} gates covering V1-V6, V9, V12")
    print(f"  Stage 2: {len(STAGE2)} gates covering V7, V8, V10, V11")
    print(f"  canvas {W} x {H} in, authored at full text width")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VALID compliance heatmap across the 74 empirical papers, regenerated from the
coding sheet.

The shipped `fig7_valid_heatmap.png` (2026-04-10) had no generator in the
repository and two defects:

  1. Layout. The Pass/Partial/Fail/N/A legend was drawn inside the heatmap axes
     and landed on top of the rotated column labels, which were themselves
     clipped at the figure edge ("V1: Class Dist.", "V2: Balance...").
  2. Item V9. Its failure-rate bar read 0%, grouped with V10 and V11. V10/V11
     are 0% because every paper is coded N/A; V9 is coded for all 74 papers and
     24 of them report no baseline beyond buy-and-hold.

Everything here derives from the coding sheet. Per-paper item states follow the
D-to-V mapping already printed in the paper's audit table (tab:audit):

    V1, V2  <- D2_class_balance      V7, V8  <- D1_cost + D6_net_perf
    V3      <- D3_temporal_split     V9      <- D5_baselines
    V4      <- D4_validation         V12     <- D7_code

Two groups are not coded per paper and are drawn as such:

    V5, V6      universal non-compliance, asserted in the text, not coded
                per paper; drawn as Fail for every row.
    V10, V11    not assessable from published text; drawn as N/A for every row.

Item failure rates use applicable-only denominators (N/A rows excluded), which
is why they differ from tab:audit -- that table reports out of n=74. V1/V2 are
95%/98% of the 58 papers where class balance applies, and 73% of all 74.

  source : results/reference/literature_audit_80.csv
  output : $FIG7_OUT (default <repo>/paper/latex/figures/fig7_valid_heatmap.pdf)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

fs.apply(base=9.0)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "reference" / "literature_audit_80.csv"
OUT = Path(os.environ.get("FIG7_OUT",
                          ROOT / "paper/latex/figures/fig7_valid_heatmap.pdf"))

# Same exclusion list as audit/audit_analysis.py, under the inclusion
# criterion stated in audit/coding_guide.md: six non-empirical entries -- two
# surveys, two synthetic-only methodological studies, one equities-only
# replication and one non-ML microstructure study -- drop out of the 80-row
# sheet, leaving 74.
NON_EMPIRICAL_IDS = [1, 9, 11, 12, 71, 79]

FAIL, PARTIAL, PASS, NA = 0, 1, 2, 3
STATE_COLORS = [fs.STATE_FAIL, fs.STATE_PARTIAL, fs.STATE_PASS, fs.STATE_NA]
STATE_LABELS = ["Fail", "Partial", "Pass", "Not applicable"]
CMAP = ListedColormap(STATE_COLORS)

ITEMS = [
    ("V1", "Class distribution"),
    ("V2", "Class balancing"),
    ("V3", "Temporal split"),
    ("V4", "CPCV with PBO"),
    ("V5", "Var(SR_IS)"),
    ("V6", "Permutation tests"),
    ("V7", "Net performance"),
    ("V8", "Cost sensitivity"),
    ("V9", "Simple baselines"),
    ("V10", "Bear markets"),
    ("V11", "Trade frequency"),
    ("V12", "Code available"),
]
CODES = [c for c, _ in ITEMS]

# Baseline strings that go beyond a bare buy-and-hold comparison.
NON_TRIVIAL_BASELINE = ["MACD", "RSI", "SMA", "momentum", "rules", "factors",
                        "random", "UBAH"]


def cell(v):
    return str(v) if pd.notna(v) else ""


def code_paper(r):
    """Map one coding-sheet row onto the twelve VALID items."""
    s = {}

    # V1/V2 <- D2. "N/A" markers cover reinforcement-learning, momentum and
    # cross-sectional designs where a directional class balance does not apply.
    bal = cell(r.D2_class_balance)
    if bal.startswith("N/A"):
        s["V1"] = s["V2"] = NA
    elif bal == "" or "Not addressed" in bal:
        s["V1"] = s["V2"] = FAIL
    elif "Partial" in bal or "Noted" in bal:
        s["V1"], s["V2"] = PARTIAL, FAIL
    else:
        s["V1"] = s["V2"] = PASS

    s["V3"] = PASS if cell(r.D3_temporal_split).startswith("Yes") else FAIL

    val = cell(r.D4_validation)
    if "CPCV" in val:
        s["V4"] = PASS
    elif "Walk-forward" in val or "Expanding" in val:
        s["V4"] = PARTIAL          # temporally sound, but no purging or PBO
    else:
        s["V4"] = FAIL

    # V5/V6 are asserted, not coded. See the module docstring.
    s["V5"] = s["V6"] = FAIL

    cost, net = cell(r.D1_cost), cell(r.D6_net_perf)
    if net.startswith("Yes"):
        s["V7"] = PASS
    elif net.startswith("Partial"):
        s["V7"] = PARTIAL
    elif cost.startswith("N/A"):
        s["V7"] = NA
    else:
        s["V7"] = FAIL

    bp = cell(r.D1_cost_bp)
    if cost == "Yes" and bp not in ("", "0", "nan", "N/A"):
        s["V8"] = PASS
    elif cost.startswith("Partial"):
        s["V8"] = PARTIAL
    elif cost.startswith("N/A"):
        s["V8"] = NA
    else:
        s["V8"] = FAIL

    base = cell(r.D5_baselines)
    only_bnh = "BnH" in base and not any(k in base for k in NON_TRIVIAL_BASELINE)
    s["V9"] = FAIL if (base == "" or only_bnh) else PASS

    s["V10"] = s["V11"] = NA

    code = cell(r.D7_code)
    if code.startswith("Yes"):
        s["V12"] = PASS
    elif code.startswith("Partial"):
        s["V12"] = PARTIAL
    else:
        s["V12"] = FAIL

    return [s[c] for c in CODES]


def short_label(authors, year):
    a = str(authors).strip()
    if len(a) > 24:
        a = a[:23].rstrip() + "."
    return f"{a} ({int(year)})"


df = pd.read_csv(SRC)
emp = df[~df["id"].isin(NON_EMPIRICAL_IDS)].copy()
assert len(emp) == 74, f"expected 74 empirical papers, got {len(emp)}"

emp = emp.sort_values(["year", "id"], kind="stable").reset_index(drop=True)
M = np.array([code_paper(r) for r in emp.itertuples()])
labels = [short_label(r.authors, r.year) for r in emp.itertuples()]

# Score: Pass 1.0, Partial 0.5, Fail and N/A 0. Out of twelve, so the two
# unassessable items cap every paper at 10 -- the same convention behind the
# 2.5/12 median the paper reports.
WEIGHT = {PASS: 1.0, PARTIAL: 0.5, FAIL: 0.0, NA: 0.0}
scores = np.array([sum(WEIGHT[v] for v in row) for row in M])
median = float(np.median(scores))

applicable = (M != NA).sum(axis=0)
fails = (M == FAIL).sum(axis=0)
fail_rate = np.divide(100.0 * fails, applicable,
                      out=np.zeros(len(CODES)), where=applicable > 0)

# Authored at the text width so LaTeX scales by 1.0 and the point sizes here
# are the point sizes on the page. Row three is an empty spacer: without it the
# legend band sat close enough to the V1-V12 labels above it to touch them.
n_papers = len(emp)
fig = plt.figure(figsize=(fs.width(1.0), 6.6))
gs = GridSpec(4, 2, figure=fig,
              width_ratios=[12.0, 3.1],
              height_ratios=[n_papers, 13.5, 5.0, 3.4],
              wspace=0.035, hspace=0.05)

# --- heatmap -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.imshow(M, cmap=CMAP, vmin=-0.5, vmax=3.5, aspect="auto",
          interpolation="nearest")

ax.set_xlim(-0.5, len(CODES) - 0.5)
ax.set_xticks(np.arange(len(CODES)))
# Column labels sit above the heatmap: the space below belongs to the
# failure-rate panel, which is where the old figure's labels collided with
# the legend. The two panels are aligned by matching xlim rather than by
# sharex, which would make them share one tick formatter and let the short
# V-codes below overwrite the full item names here.
ax.xaxis.set_ticks_position("top")
ax.set_xticklabels([f"{c}: {n}" for c, n in ITEMS], rotation=45, ha="left",
                   rotation_mode="anchor", fontsize=7.2)
ax.tick_params(axis="x", length=0, pad=2)

ax.set_yticks(np.arange(n_papers))
ax.set_yticklabels(labels, fontsize=4.6)
ax.tick_params(axis="y", length=0, pad=1.5)

ax.set_xticks(np.arange(-0.5, len(CODES), 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_papers, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.35)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_visible(False)

# --- per-paper score -----------------------------------------------------
ax_s = fig.add_subplot(gs[0, 1], sharey=ax)
ax_s.barh(np.arange(n_papers), scores, height=0.72, color=fs.FILL_MID,
          edgecolor=fs.STROKE, linewidth=0.25)
ax_s.axvline(median, color=fs.ACCENT, linewidth=1.0, linestyle="--", zorder=5)
ax_s.set_xlim(0, 12)
ax_s.set_xticks([0, 3, 6, 9, 12])
ax_s.tick_params(axis="x", labelsize=7.5, pad=1.5)
# Axis at the bottom: the space above this panel belongs to the heatmap's
# rotated column labels, and "V12: Code available" ran straight into a top
# axis label. The median is keyed here rather than in a cell of its own.
ax_s.set_xlabel(f"VALID score (of 12)\ndashed: median = {median:.1f}",
                fontsize=7.5, labelpad=4, linespacing=1.4)
plt.setp(ax_s.get_yticklabels(), visible=False)
ax_s.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax_s.spines[s].set_visible(False)
ax_s.grid(axis="x", color=fs.GRIDLINE, linewidth=0.4, zorder=0)
ax_s.set_axisbelow(True)

# --- failure rate per item ----------------------------------------------
ax_b = fig.add_subplot(gs[1, 0])
bar_colors = [STATE_COLORS[NA] if applicable[i] == 0 else STATE_COLORS[FAIL]
              for i in range(len(CODES))]
ax_b.bar(np.arange(len(CODES)), fail_rate, width=0.66, color=bar_colors,
         edgecolor=fs.GRAY_STROKE, linewidth=0.3)
for i, v in enumerate(fail_rate):
    txt = "n/a" if applicable[i] == 0 else f"{v:.0f}%"
    ax_b.text(i, v + 3, txt, ha="center", va="bottom", fontsize=6.8,
              fontweight="bold", color=fs.INK_DEEP)
ax_b.set_ylim(0, 118)
ax_b.set_yticks([0, 50, 100])
ax_b.set_ylabel("Fail (%)", fontsize=8)
ax_b.tick_params(axis="y", labelsize=7.5, pad=1.5)
ax_b.set_xticks(np.arange(len(CODES)))
ax_b.set_xticklabels(CODES, fontsize=8)
ax_b.tick_params(axis="x", labelsize=8, length=2, pad=3, top=False,
                 labeltop=False, bottom=True, labelbottom=True)
ax_b.set_xlim(-0.5, len(CODES) - 0.5)
for s in ("top", "right"):
    ax_b.spines[s].set_visible(False)
ax_b.grid(axis="y", color=fs.GRIDLINE, linewidth=0.4, zorder=0)
ax_b.set_axisbelow(True)

# --- legend, in a row of its own ----------------------------------------
ax_l = fig.add_subplot(gs[3, :])
ax_l.axis("off")
ax_l.legend(handles=[Patch(facecolor=STATE_COLORS[i], edgecolor="#555555",
                           linewidth=0.3, label=STATE_LABELS[i])
                     for i in (PASS, PARTIAL, FAIL, NA)],
            loc="center", ncol=4, fontsize=8.5, frameon=False,
            handlelength=1.5, handleheight=0.9, columnspacing=2.2)

fs.save(fig, OUT)
print(f"Saved {OUT}")
print(f"  papers={n_papers}, median score={median:.2f}/12, "
      f"mean={scores.mean():.2f}, range=[{scores.min():.1f}, {scores.max():.1f}]")
print("  fail rate (applicable-only denominator):")
for i, (c, name) in enumerate(ITEMS):
    if applicable[i] == 0:
        print(f"    {c:<4}{name:<22} n/a (all {n_papers} coded N/A)")
    else:
        print(f"    {c:<4}{name:<22} {fails[i]:>2}/{applicable[i]:<2} "
              f"= {fail_rate[i]:>3.0f}%")

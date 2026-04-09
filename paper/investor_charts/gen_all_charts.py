#!/usr/bin/env python3
"""
Alpha-33 Investor Report — Hedge Fund Quality Charts
=====================================================
4 charts for Notion embedding via GitHub raw URLs.
Dark theme, institutional style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ── Global style: dark institutional ─────────────────────────────
BG = "#0D1117"
FG = "#C9D1D9"
ACCENT = "#58A6FF"
GREEN = "#3FB950"
RED = "#F85149"
GOLD = "#D29922"
GRID = "#21262D"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "savefig.facecolor": BG,
})

# ── Load data ────────────────────────────────────────────────────
yearly = pd.read_csv("/Users/jaewookkim/jwquant/results/paper_stats/yearly_v2_vs_bnh.csv")
yearly = yearly[yearly["year"] != "Full"]
yearly["year"] = yearly["year"].astype(int)
yearly = yearly[yearly["year"] <= 2025]  # exclude partial 2026

regime = pd.read_csv("/Users/jaewookkim/jwquant/results/paper_stats/regime_performance.csv")

# ================================================================
# CHART 1: Cumulative Growth ($100)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

cum_a33 = 100 * (1 + yearly["v2_return"]).cumprod()
cum_bnh = 100 * (1 + yearly["bnh_return"]).cumprod()

# Insert starting point
years = [yearly["year"].iloc[0] - 1] + yearly["year"].tolist()
vals_a33 = [100] + cum_a33.tolist()
vals_bnh = [100] + cum_bnh.tolist()

ax.fill_between(years, vals_a33, alpha=0.15, color=ACCENT)
ax.plot(years, vals_a33, color=ACCENT, linewidth=2.5, label="Alpha-33 v4", marker="o", markersize=5)
ax.plot(years, vals_bnh, color=RED, linewidth=1.8, label="BTC Buy & Hold", marker="s", markersize=4, alpha=0.8)
ax.axhline(100, color=FG, alpha=0.2, linewidth=0.8, linestyle="--")

# Annotate final values
ax.annotate(f"${vals_a33[-1]:,.0f}", xy=(years[-1], vals_a33[-1]),
            xytext=(15, 10), textcoords="offset points",
            color=ACCENT, fontsize=13, fontweight="bold")
ax.annotate(f"${vals_bnh[-1]:,.0f}", xy=(years[-1], vals_bnh[-1]),
            xytext=(15, -15), textcoords="offset points",
            color=RED, fontsize=11)

ax.set_title("Growth of $100  (2018 – 2025)", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("Portfolio Value ($)")
ax.set_xlabel("")
ax.legend(loc="upper left", fontsize=11, framealpha=0.3, edgecolor=GRID)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_xticks(years)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT / "chart1_equity_curve.png")
plt.close()
print("✓ Chart 1: Equity curve")

# ================================================================
# CHART 2: Annual Returns Bar Chart (Alpha-33 vs BTC B&H)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(yearly))
w = 0.35

bars1 = ax.bar(x - w/2, yearly["v2_return"] * 100, w, color=ACCENT, label="Alpha-33", zorder=3, edgecolor=BG, linewidth=0.5)
bars2 = ax.bar(x + w/2, yearly["bnh_return"] * 100, w, color=RED, alpha=0.7, label="BTC B&H", zorder=3, edgecolor=BG, linewidth=0.5)

ax.axhline(0, color=FG, alpha=0.3, linewidth=0.8)

# Highlight alpha years
for i, row in yearly.iterrows():
    idx = yearly.index.get_loc(i)
    alpha = row["alpha"] * 100
    if alpha > 10:
        ax.annotate(f"+{alpha:.0f}%α", xy=(idx, max(row["v2_return"], row["bnh_return"]) * 100 + 8),
                    ha="center", fontsize=8, color=GREEN, fontweight="bold")

ax.set_title("Annual Returns: Alpha-33 vs BTC Buy & Hold", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("Return (%)")
ax.set_xticks(x)
ax.set_xticklabels(yearly["year"].astype(str))
ax.legend(loc="upper left", fontsize=11, framealpha=0.3, edgecolor=GRID)
ax.grid(True, axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT / "chart2_annual_returns.png")
plt.close()
print("✓ Chart 2: Annual returns")

# ================================================================
# CHART 3: Drawdown Comparison
# ================================================================
fig, ax = plt.subplots(figsize=(10, 4.5))

dd_a33 = yearly["v2_mdd"].values * 100
dd_bnh = yearly["bnh_mdd"].values * 100

ax.bar(x - w/2, dd_a33, w, color=ACCENT, label="Alpha-33 MDD", zorder=3, edgecolor=BG)
ax.bar(x + w/2, dd_bnh, w, color=RED, alpha=0.7, label="BTC B&H MDD", zorder=3, edgecolor=BG)
ax.axhline(0, color=FG, alpha=0.2, linewidth=0.8)

# Annotate worst
worst_bnh_idx = np.argmin(dd_bnh)
ax.annotate(f"{dd_bnh[worst_bnh_idx]:.0f}%", xy=(worst_bnh_idx + w/2, dd_bnh[worst_bnh_idx]),
            xytext=(10, -15), textcoords="offset points",
            color=RED, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))

ax.set_title("Maximum Drawdown by Year", fontsize=16, fontweight="bold", pad=15)
ax.set_ylabel("Drawdown (%)")
ax.set_xticks(x)
ax.set_xticklabels(yearly["year"].astype(str))
ax.legend(loc="lower left", fontsize=11, framealpha=0.3, edgecolor=GRID)
ax.grid(True, axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT / "chart3_drawdown.png")
plt.close()
print("✓ Chart 3: Drawdown")

# ================================================================
# CHART 4: Return Decomposition (v4 verified, 2022-2025)
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5))

components = ["BTC\nDirectional", "Rebalancing\nPremium", "Carry\n(FR+Earn)", "DM Regime\nShift"]
values = [8.2, 1.7, 5.3, 3.4]
colors = [ACCENT, GREEN, GOLD, "#A371F7"]
total = sum(values)

bars = ax.barh(components[::-1], values[::-1], color=colors[::-1], height=0.6,
               edgecolor=BG, linewidth=1, zorder=3)

for bar, val in zip(bars, values[::-1]):
    pct = val / total * 100
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"+{val}%  ({pct:.0f}%)", va="center", fontsize=11, color=FG, fontweight="bold")

ax.set_title(f"Return Decomposition — v4 (2022–2025)\nTotal CAGR: +{total:.1f}%",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Annual Contribution (%p)")
ax.set_xlim(0, max(values) + 4)
ax.grid(True, axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT / "chart4_decomposition.png")
plt.close()
print("✓ Chart 4: Return decomposition")

# ================================================================
# CHART 5: Risk-Return Scatter (positioning)
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5.5))

strategies = {
    "Alpha-33 v4\n(2022-25)": (1.06, 20.9, ACCENT, 200),
    "BTC B&H": (0.69, 66.8, RED, 120),
    "S&P 500": (0.90, 25.4, "#8B949E", 100),
    "60/40": (0.65, 22.0, "#8B949E", 80),
    "Static 30%\nRebal+Carry": (1.09, 19.6, GREEN, 140),
}

for name, (sr, mdd, color, size) in strategies.items():
    ax.scatter(mdd, sr, s=size, c=color, zorder=5, edgecolors="white", linewidth=1.5)
    offset = (10, 5) if "Alpha" not in name else (12, -15)
    ax.annotate(name, xy=(mdd, sr), xytext=offset, textcoords="offset points",
                fontsize=9, color=color, fontweight="bold" if "Alpha" in name or "Static" in name else "normal")

ax.axhline(1.0, color=GOLD, alpha=0.4, linewidth=1, linestyle="--", label="SR = 1.0 threshold")
ax.set_xlabel("Maximum Drawdown (%)", fontsize=12)
ax.set_ylabel("Sharpe Ratio", fontsize=12)
ax.set_title("Risk-Return Positioning", fontsize=16, fontweight="bold", pad=15)
ax.invert_xaxis()
ax.legend(loc="lower left", fontsize=9, framealpha=0.3, edgecolor=GRID)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT / "chart5_risk_return.png")
plt.close()
print("✓ Chart 5: Risk-return scatter")

print(f"\n✓ All 5 charts saved to {OUT}")

#!/usr/bin/env python3
"""
KDD-MLF 2026 · Paper 18 — VALID poster, 900 x 1050 mm.

Reflow of gen_poster.py for the assigned board (3 ft x 3.5 ft = 914 x 1067 mm).
The 900 x 1600 original overruns that height by 533 mm.

Geometry only. Every string, every number, the palette and Inter are carried
over unchanged from gen_poster.py — nothing is recomputed and nothing is
reworded. The original file is kept as the record of the 900 x 1600 build.

What moved: the six stacked tiers become two content bands on a four-column
grid, the two contrast cards stack instead of sitting side by side, "Who is
this for" drops to one line per persona, and the reference list is cut to
three. Everything on the must-keep list is intact at full size.

  python3 gen_poster_900x1050.py            -> exact 900 x 1050
  python3 gen_poster_900x1050.py --bleed    -> 906 x 1056 (3 mm bleed)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrow
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.textpath import TextPath
import segno

# ----------------------------------------------------------------- fonts
FONT_DIR = os.environ.get(
    "INTER_TTF_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "assets", "fonts", "inter"))
FACES = {}
for w, fn in [("R", "Inter-Regular.ttf"), ("M", "Inter-Medium.ttf"),
              ("SB", "Inter-SemiBold.ttf"), ("B", "Inter-Bold.ttf"),
              ("EB", "Inter-ExtraBold.ttf"), ("BK", "Inter-Black.ttf"),
              ("I", "Inter-Italic.ttf"), ("MI", "Inter-MediumItalic.ttf")]:
    p = os.path.join(FONT_DIR, fn)
    if os.path.exists(p):
        fontManager.addfont(p)
        FACES[w] = FontProperties(fname=p)
if not FACES:
    for w in ["R", "M", "SB", "B", "EB", "BK", "I", "MI"]:
        FACES[w] = FontProperties(family="Helvetica")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 6

# ----------------------------------------------------------------- palette
INK      = "#13181D"
INK_SOFT = "#4A555F"
RULE     = "#C9D1D6"
GREEN    = "#0E5C42"
GREEN_D  = "#083A2A"
GREEN_L  = "#E4F0EA"
RED      = "#B02418"
RED_L    = "#FBEAE7"
PAPER    = "#FFFFFF"
SLATE    = "#1E2A32"

BLEED = 3.0 if "--bleed" in sys.argv else 0.0
W, H = 900.0, 1050.0
L, R = 45.0, 855.0                    # 45 mm pin margin, well past the 15 mm floor
CW = R - L                            # 810

# four-column grid
GUT = 18.0
COLW = (CW - 3 * GUT) / 4             # 189
CX = [L + i * (COLW + GUT) for i in range(4)]
# the money block spans columns 2-3
MIDX = CX[1]
MIDW = COLW * 2 + GUT                 # 396

URL_REPO = "https://github.com/orcajae/valid-framework"
URL_SSRN = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6508779"
URL_LANDING = "https://orcajae.github.io/valid-framework/"

fig = plt.figure(figsize=((W + 2 * BLEED) / 25.4, (H + 2 * BLEED) / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-BLEED, W + BLEED)
ax.set_ylim(H + BLEED, -BLEED)        # y grows downward
ax.axis("off")
ax.add_patch(Rectangle((-BLEED, -BLEED), W + 2 * BLEED, H + 2 * BLEED,
                       fc=PAPER, ec="none", zorder=-10))


# ----------------------------------------------------------------- helpers
def tw(s, size, w="R"):
    if not s.strip():
        return 0.0
    return TextPath((0, 0), s, size=size,
                    prop=FACES[w]).get_extents().width / 72 * 25.4


def T(x, y, s, size=30, w="R", color=INK, ha="left", va="top", zorder=5, **kw):
    return ax.text(x, y, s, fontproperties=FACES[w], fontsize=size, color=color,
                   ha=ha, va=va, zorder=zorder, **kw)


def wrap(s, max_mm, size, w="R"):
    out, line = [], ""
    for word in s.split():
        trial = (line + " " + word).strip()
        if tw(trial, size, w) <= max_mm or not line:
            line = trial
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def para(x, y, s, max_mm, size=30, w="R", color=INK, lead=1.30, ha="left",
         zorder=5):
    lh = size / 72 * 25.4 * lead
    lines = wrap(s, max_mm, size, w)
    for i, ln in enumerate(lines):
        T(x, y + i * lh, ln, size, w, color, ha=ha, zorder=zorder)
    return y + len(lines) * lh


def fit(s, max_mm, size, w="R"):
    while size > 6 and tw(s, size, w) > max_mm:
        size -= 1
    return size


def box(x, y, bw, bh, fc, ec="none", lw=0, r=4.0, z=1):
    ax.add_patch(FancyBboxPatch((x + r, y + r), bw - 2 * r, bh - 2 * r,
                                boxstyle=f"round,pad={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, mutation_aspect=1))


def rule(x, y, length, lw=1.2, color=RULE, z=2):
    ax.plot([x, x + length], [y, y], lw=lw, color=color, zorder=z,
            solid_capstyle="butt")


def qr(x, y, size_mm, data, fg=INK, quiet=3, z=6):
    q = segno.make(data, error="m")
    m = [list(row) for row in q.matrix]
    n = len(m)
    u = size_mm / (n + 2 * quiet)
    ax.add_patch(Rectangle((x, y), size_mm, size_mm, fc="white", ec="none",
                           zorder=z))
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            if v:
                ax.add_patch(Rectangle((x + (j + quiet) * u,
                                        y + (i + quiet) * u),
                                       u * 1.03, u * 1.03, fc=fg, ec="none",
                                       lw=0, zorder=z + 1))


def cax(x, y, w_, h_):
    a = fig.add_axes([(x + BLEED) / (W + 2 * BLEED),
                      1 - (y + h_ + BLEED) / (H + 2 * BLEED),
                      w_ / (W + 2 * BLEED), h_ / (H + 2 * BLEED)])
    for s in a.spines.values():
        s.set_color(RULE)
        s.set_linewidth(1.0)
    a.tick_params(colors=INK_SOFT, labelsize=17, length=3, width=1.0)
    for lb in a.get_xticklabels() + a.get_yticklabels():
        lb.set_fontproperties(FACES["M"])
    return a


def checkbox(x, y, s=9.0, lw=1.6, color=INK_SOFT, z=6):
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=color, lw=lw, zorder=z))


# =================================================================
# HEADER  (0–104 mm)
# =================================================================
ax.add_patch(Rectangle((-BLEED, -BLEED), W + 2 * BLEED, 10 + BLEED, fc=GREEN,
                       ec="none", zorder=3))

t1 = "VALID: A 12-Item Validation Checklist"
t2 = "for Financial Machine Learning"
TS = min(fit(t1, CW - 130, 82, "EB"), fit(t2, CW - 130, 82, "EB"))
T(L, 26, t1, TS, "EB", INK)
T(L, 26 + TS / 72 * 25.4 * 1.06, t2, TS, "EB", INK)

# Academic attribution line under the formal title: 24 pt = x-height 4.62 mm,
# readable at 1.5 m. The hook banner below keeps its 72 pt dominance.
# The e-mail address is carried once, in the footer, not here.
byline = ("Jaewook Kim  ·  Independent Researcher  ·  "
          "KDD-MLF 2026, Jeju  ·  Paper 18 (Oral)")
T(L, 86, byline, fit(byline, 640, 24, "M"), "M", INK_SOFT)

qr(741, 22, 46, URL_SSRN)
T(764, 71, "SSRN 6508779", 17, "SB", INK_SOFT, ha="center")
qr(801, 22, 46, URL_REPO)
T(824, 71, "GitHub repo", 17, "SB", INK_SOFT, ha="center")

# =================================================================
# HOOK BANNER  (106–188 mm)
# =================================================================
ax.add_patch(Rectangle((-BLEED, 106), W + 2 * BLEED, 82, fc=GREEN, ec="none",
                       zorder=1))
h1, h2 = "Statistically perfect.", "Economically worthless."
HS = min(fit(h1, CW / 2 - 30, 72, "BK"), fit(h2, CW / 2 - 30, 72, "BK"))
T(W / 2 - 14, 118, h1, HS, "BK", "#8FCFB4", ha="right", zorder=4)
T(W / 2 + 14, 118, h2, HS, "BK", "#FFFFFF", ha="left", zorder=4)
sub = ("340 strategy variants  ·  80 papers audited  ·  "
       "one 12-item gate that tells them apart in ~5 minutes")
T(W / 2, 160, sub, fit(sub, CW - 60, 26, "M"), "M", "#D6EFE3", ha="center",
  zorder=4)

# =================================================================
# BAND A  (200–556 mm) — field today · money block · the gate
# =================================================================
AY = 204
for x_, w_, ttl in [(CX[0], COLW, "The Field Today"),
                    (MIDX, MIDW, "The Strategies That Passed Everything"),
                    (CX[3], COLW, "The Gate")]:
    T(x_, AY, ttl, fit(ttl, w_, 34, "B"), "B", GREEN)
    rule(x_, AY + 17, w_, lw=2.0, color=GREEN)

# ---------------------------------------------- left: the field today
FY = 228
T(CX[0], FY, "2.5", 76, "BK", RED)
T(CX[0] + tw("2.5", 76, "BK") + 8, FY + 16, "of 12", 26, "B", INK)
T(CX[0] + tw("2.5", 76, "BK") + 8, FY + 34, "items met", 22, "M", INK_SOFT)
T(CX[0], FY + 56, "median audited paper (n = 74 empirical, 2018–2026)", 18, "M",
  INK_SOFT)
T(CX[0], FY + 76, "Share of papers failing each dimension", 20, "SB", INK_SOFT)

BAR0, BARW = CX[0] + 96.0, 62.0
rows = [("No code", 85), ("No class balance", 73), ("Costs omitted", 54),
        ("No net performance", 54), ("BnH-only baseline", 35),
        ("Random split", 7), ("CPCV used", 0)]
for i, (lab, val) in enumerate(rows):
    yy = FY + 92 + i * 17
    T(BAR0 - 5, yy - 1, lab, fit(lab, 88, 24, "M"), "M", INK, ha="right")
    ax.add_patch(Rectangle((BAR0, yy), BARW, 10, fc="#EDF1F3", ec="none",
                           zorder=2))
    if val > 0:
        ax.add_patch(Rectangle((BAR0, yy), BARW * val / 100, 10,
                               fc=GREEN if val < 60 else GREEN_D, ec="none",
                               zorder=3))
    T(CX[0] + COLW, yy - 3, f"{val}%", 24, "EB", RED if val == 0 else INK,
      ha="right")
para(CX[0], FY + 218, "This is the reporting norm, not a few bad papers.",
     COLW, 22, "MI", INK)

# ---------------------------------------------- centre: contrast cards
CY, CH = 228, 128
cw_ = MIDW / 2 - 8
# PASSED
box(MIDX, CY, cw_, CH, GREEN_L, ec=GREEN, lw=2.0)
ax.add_patch(Rectangle((MIDX, CY), cw_, 26, fc=GREEN, ec="none", zorder=2))
T(MIDX + 10, CY + 5, "PASSED", 24, "BK", "#FFFFFF", zorder=4)
T(MIDX + cw_ - 10, CY + 7, "9 of 340 variants", 24, "SB", "#BFE3D2", ha="right",
  zorder=4)
for i, (v, k) in enumerate([("Bonferroni: 9", "Holm 9 · BH-FDR 10"),
                            ("t up to 21.7", "Harvey t > 3.0: 10 survive"),
                            ("SR up to 1.98", "vs benchmark 0.917")]):
    yy = CY + 34 + i * 28
    T(MIDX + 10, yy, v, 28, "EB", GREEN_D, zorder=4)
    T(MIDX + 10, yy + 12, k, 24, "M", INK_SOFT, zorder=4)
T(MIDX + 10, CY + CH - 15, "Every classical correction: PASS", 24, "MI",
  GREEN_D, zorder=4)

# REALITY
rx = MIDX + cw_ + 16
box(rx, CY, cw_, CH, RED_L, ec=RED, lw=2.0)
ax.add_patch(Rectangle((rx, CY), cw_, 26, fc=RED, ec="none", zorder=2))
T(rx + 10, CY + 5, "REALITY", 24, "BK", "#FFFFFF", zorder=4)
T(rx + cw_ - 10, CY + 7, "the same nine", 24, "SB", "#F2C6C0", ha="right",
  zorder=4)
for i, (v, k) in enumerate([("PBO = 1.0", "all nine — complete overfitting"),
                            ("AUC 0.47–0.64", "near random"),
                            ("DSR: 0 of 340", "E[max SR] under null 2.93")]):
    yy = CY + 34 + i * 28
    T(rx + 10, yy, v, 28, "EB", RED, zorder=4)
    T(rx + 10, yy + 12, k, 24, "M", INK_SOFT, zorder=4)
T(rx + 10, CY + CH - 15, "Corrections cannot see overfitting", 24, "MI", RED,
  zorder=4)

T(MIDX, CY + CH + 6,
  "Multiple-testing control is necessary — and demonstrably not sufficient.",
  21, "MI", INK_SOFT, zorder=4)

# dark evidence strip
SX, SH = CY + CH + 26, 40
box(MIDX, SX, MIDW, SH, SLATE, r=5, z=2)
s1 = ("Across all 340 variants: 52% lose money after costs · "
      "only 4.4% beat the benchmark.")
s2 = ("All nine trade daily. CPCV rates every one fully overfit; "
      "the DSR rejects all 340.")
T(MIDX + 13, SX + 7, s1, fit(s1, MIDW - 26, 24, "SB"), "SB", "#FFFFFF",
  zorder=4)
T(MIDX + 13, SX + 23, s2, fit(s2, MIDW - 26, 24, "SB"), "SB", "#9FD8BF",
  zorder=4)

# bull bias moves under the money block
BBY = SX + SH + 16
T(MIDX, BBY, "Bull Bias", 26, "B", INK)
T(MIDX + tw("Bull Bias", 26, "B") + 10, BBY + 5,
  "structural class imbalance (V1–V2)", 19, "M", INK_SOFT)
bb = cax(MIDX + 4, BBY + 24, MIDW / 2 - 14, 96)
labels = ["BTC\nCatBoost", "ETH\nCatBoost", "SOL\nCatBoost", "BTC\nLSTM"]
unb = [97.2, 90.5, 97.0, 57.7]
bal = [42.3, 45.2, 32.2, 52.3]
xs = range(len(labels))
bb.bar([i - 0.20 for i in xs], unb, width=0.38, color=RED)
bb.bar([i + 0.20 for i in xs], bal, width=0.38, color=GREEN)
bb.axhline(50, color=INK_SOFT, ls=(0, (4, 3)), lw=1.2)
bb.set_xticks(list(xs))
bb.set_xticklabels(labels)
bb.set_ylim(0, 105)
bb.set_yticks([0, 50, 100])
bb.set_ylabel('predicted "long" (%)', fontproperties=FACES["M"], fontsize=16,
              color=INK_SOFT)
bb.spines["top"].set_visible(False)
bb.spines["right"].set_visible(False)

bx = MIDX + MIDW / 2 + 6
y_ = BBY + 26
y_ = para(bx, y_, "Tree models predict “long” 90–97% of the time. After "
                  "balancing, AUC converges to ≈0.50 — a coin flip.",
          MIDW / 2 - 6, 22, "SB", INK)
y_ += 6
for lab, val, col in [("Balanced AUC (BTC / ETH / SOL)", "0.50 · 0.49 · 0.51",
                       INK),
                      ("Net SR, 1h balanced", "−0.88 · −1.28 · −0.21", RED)]:
    T(bx, y_, lab, fit(lab, 118, 19, "M"), "M", INK_SOFT)
    T(bx + MIDW / 2 - 6, y_ - 1, val, 19, "B", col, ha="right")
    rule(bx, y_ + 10, MIDW / 2 - 6, lw=0.8)
    y_ += 14
T(bx, y_ + 1, "All 1h models: PBO = 1.000.", 20, "SB", RED)

# ---------------------------------------------- right: the gate
# T2 promotion: the two stage cards carry the item names at 19 pt
# (x-height 3.66 mm) instead of 12 pt, so the box grows 32 -> 44 mm and the
# lines wrap rather than shrink. Diagram area +37%.
GY = 228
GBH, GSTEP = 44.0, 58.0
for k, (ttl, desc) in enumerate([
        ("Stage 1 — Statistical",
         "V1–V6 · balance, temporal split, CPCV+PBO, variance, permutation"),
        ("Stage 2 — Economic",
         "V7–V12 · net costs, cost sweep, baselines, regimes, turnover, code")]):
    gy = GY + k * GSTEP
    box(CX[3], gy, COLW, GBH, GREEN_L, ec=GREEN, lw=1.6)
    T(CX[3] + 9, gy + 6, ttl, 22, "B", GREEN_D, zorder=4)
    for i, ln in enumerate(wrap(desc, COLW - 18, 19, "M")):
        T(CX[3] + 9, gy + 22 + i * 8.5, ln, 19, "M", INK_SOFT, zorder=4)
ax.add_patch(FancyArrow(CX[3] + COLW / 2, GY + GBH + 2, 0, 8, width=2.0,
                        head_width=8, head_length=4, fc=GREEN, ec="none",
                        zorder=4, length_includes_head=True))
T(CX[3], GY + GSTEP + GBH + 4, "Binary items · fixed order · pass/fail", 18,
  "MI", INK_SOFT)

MY = GY + GSTEP + GBH + 24
T(CX[3], MY, "False positives on signal-free data (n = 200)", 18, "SB", INK)
for i, (lab, val, ci, col) in enumerate([
        ("AUC alone", "27%", "[21, 34]", RED),
        ("+ CPCV & PBO", "0%", "[0, 1.9]", GREEN)]):
    yy = MY + 16 + i * 20
    T(CX[3], yy, lab, 21, "M", INK)
    T(CX[3] + COLW - 38, yy - 4, val, 26, "EB", col, ha="right")
    T(CX[3] + COLW, yy + 1, ci, 17, "M", INK_SOFT, ha="right")
T(CX[3], MY + 58, "One item (V4) does it — used by 0 of 74.", 19, "SB", GREEN_D)

QY = MY + 80
box(CX[3], QY, COLW, 96, "#F1F6F3", ec=GREEN, lw=2.4)
qr(CX[3] + 8, QY + 8, 80, URL_LANDING)
tx = CX[3] + 95
T(tx, QY + 10, "Run the", 25, "B", GREEN_D, zorder=4)
T(tx, QY + 26, "checklist on", 25, "B", GREEN_D, zorder=4)
T(tx, QY + 42, "your own", 25, "B", GREEN_D, zorder=4)
T(tx, QY + 58, "strategy", 25, "B", GREEN_D, zorder=4)
T(tx, QY + 76, "~5 minutes", 22, "BK", RED, zorder=4)
T(CX[3], QY + 100, "orcajae.github.io/valid-framework", 16, "M", INK_SOFT,
  zorder=4)

# =================================================================
# BAND B  (572–930 mm) — cost illusion · the checklist · self-audit
# =================================================================
# The bull-bias x tick labels reach y 568, so the band rule sits below them.
rule(L, 574, CW, lw=2.0, color=RULE)
BY = 588
T(CX[0], BY, "Cost Illusion", 26, "B", INK)
# Its own line: at 17 pt this subtitle ran to x 258 and collided with the
# centre column's heading at x 252.
T(CX[0], BY + 20, "net-of-cost performance by frequency (V7–V8, V11)", 19, "M",
  INK_SOFT)

cb = cax(CX[0] + 4, BY + 38, COLW - 8, 128)
TFD = [("15m", -1.798, 9, 100), ("1h", -1.323, 133, 96),
       ("4h", 0.276, 9, 44), ("1d", 0.541, 77, 22)]
xs = range(len(TFD))
cb.bar(list(xs), [d[1] for d in TFD], width=0.62,
       color=[RED if d[1] < 0 else GREEN for d in TFD])
for i, d in enumerate(TFD):
    va_ = "top" if d[1] < 0 else "bottom"
    off = -0.09 if d[1] < 0 else 0.09
    cb.text(i, d[1] + off, f"{d[1]:+.2f}", ha="center", va=va_,
            fontproperties=FACES["B"], fontsize=16,
            color=RED if d[1] < 0 else GREEN)
cb.axhline(0, color=INK, lw=1.2)
cb.set_xticks(list(xs))
cb.set_xticklabels([f"{d[0]}\nn={d[2]}" for d in TFD])
cb.set_ylim(-2.35, 1.05)
cb.set_yticks([-2, -1, 0, 1])
cb.set_ylabel("median net Sharpe, 18 bp", fontproperties=FACES["M"],
              fontsize=15, color=INK_SOFT)
cb.spines["top"].set_visible(False)
cb.spines["right"].set_visible(False)

y_ = BY + 182
for lab, fr in [("", 0), ("median", 0.60), ("best", 0.80), ("< 0", 1.0)]:
    T(CX[0] + COLW * fr, y_, lab, 18, "SB", INK_SOFT,
      ha="left" if fr == 0 else "right")
y_ += 20
for tf, med, best, neg in [("15m  (n = 9)", "−1.80", "−1.33", "100%"),
                           ("1h  (n = 133)", "−1.32", "+0.32", "96%"),
                           ("4h  (n = 9)", "+0.28", "+0.57", "44%"),
                           ("1d  (n = 77)", "+0.54", "+1.98", "22%")]:
    T(CX[0], y_, tf, 19, "M", INK)
    for val, fr in [(med, 0.60), (best, 0.80), (neg, 1.0)]:
        T(CX[0] + COLW * fr, y_ - 1, val, 20, "B",
          RED if val.startswith("−") or val == "100%" else INK, ha="right")
    rule(CX[0], y_ + 13, COLW, lw=0.8)
    y_ += 21
T(CX[0], y_ + 6, "One 1h CatBoost run: gross SR 0.750 → net 0.135 at 18 bp",
  fit("One 1h CatBoost run: gross SR 0.750 → net 0.135 at 18 bp", COLW, 19,
      "SB"), "SB", RED)
T(CX[0], y_ + 24, "— 82% of gross Sharpe consumed at 101 trades / yr.",
  fit("— 82% of gross Sharpe consumed at 101 trades / yr.", COLW, 19, "SB"),
  "SB", RED)
T(CX[0], y_ + 46, "340-variant corpus, 18 bp round-trip, model variants only.",
  fit("340-variant corpus, 18 bp round-trip, model variants only.", COLW, 16,
      "M"), "M", INK_SOFT)

# ---------------------------------------------- checklist (cols 2-3)
T(MIDX, BY, "Score the last paper you reviewed", 30, "B", INK)
cbx = MIDX + tw("Score the last paper you reviewed", 30, "B") + 12
for i in range(12):
    checkbox(cbx + i * 11, BY + 3, s=9)
T(MIDX, BY + 22, "median in our audit: 2.5 / 12", 21, "SB", RED)

ITEMS = [("V1", "Report prediction class distribution"),
         ("V2", "Test with / without class balancing"),
         ("V3", "Use temporal splitting only"),
         ("V4", "Apply CPCV with PBO"),
         ("V5", "Report Var(SR_IS)"),
         ("V6", "Include permutation tests (≥100)"),
         ("V7", "Report net performance with costs"),
         ("V8", "Cost sensitivity analysis"),
         ("V9", "Compare against simple baselines"),
         ("V10", "Evaluate in bear markets and regime transitions"),
         ("V11", "Report trade frequency"),
         ("V12", "Provide code for reproducibility")]
CLX = [MIDX, MIDX + MIDW / 2 + 4]
NAMEW = MIDW / 2 - 52
CL_LEAD, CL_GAP = 13.0, 11.0
# 24 pt is the floor here, so a name that will not fit the column wraps rather
# than shrinking; the row grows for both columns so they stay aligned.
CL_LINES = [wrap(name, NAMEW, 24, "M") for _, name in ITEMS]
CL_Y, y_ = [], BY + 52
for row in range(6):
    CL_Y.append(y_)
    y_ += max(len(CL_LINES[c * 6 + row]) for c in range(2)) * CL_LEAD + CL_GAP
for i, (tag, name) in enumerate(ITEMS):
    col, row = i // 6, i % 6
    x_, ry = CLX[col], CL_Y[row]
    checkbox(x_, ry - 1, s=9)
    T(x_ + 14, ry - 3, tag, 24, "EB", GREEN)
    for j, ln in enumerate(CL_LINES[i]):
        T(x_ + 42, ry - 3 + j * CL_LEAD, ln, 24, "M", INK)
T(MIDX, y_ + 6,
  "V1–V6  Stage 1 (statistical)          V7–V12  Stage 2 (economic)", 24, "SB",
  INK_SOFT)

# self-audit
HY = BY + 250
box(MIDX, HY, MIDW, 78, "#FFFFFF", ec=RED, lw=2.4)
T(MIDX + 14, HY + 9, "Self-audit of this paper", 25, "B", RED, zorder=4)
T(MIDX + 14, HY + 27, "9 / 12", 42, "BK", INK, zorder=4)
T(MIDX + 14 + tw("9 / 12", 42, "BK") + 8, HY + 41, "fully satisfied", 20, "M",
  INK_SOFT, zorder=4)
T(MIDX + 150, HY + 30, "Three partials disclosed: V5 (flatness criteria "
  "disagree),", 24, "M", INK, zorder=4)
T(MIDX + 150, HY + 44, "V8 (single 18 bp cost level), V10 (3 crypto assets "
  "only).", 24, "M", INK, zorder=4)
T(MIDX, HY + 84, "v1.0 — not a certification. The checklist has had no formal "
  "Delphi review; community revision is invited.",
  fit("v1.0 — not a certification. The checklist has had no formal Delphi "
      "review; community revision is invited.", MIDW, 19, "MI"), "MI", INK_SOFT)

# ---------------------------------------------- who is this for (col 4)
T(CX[3], BY, "Who Is This For?", 26, "B", INK)
T(CX[3], BY + 22, "adoption paths", 19, "M", INK_SOFT)
y_ = BY + 52
for ttl, one in [("Reviewers & PCs",
                  "A minimum reporting standard for financial ML submissions."),
                 ("Validation & model-risk teams",
                  "A two-stage gate for incoming strategies, both binary."),
                 ("Allocators & operational due diligence",
                  "Twelve questions before trusting a pitched Sharpe ratio.")]:
    lines = wrap(one, COLW - 20, 19, "M")
    bh = 26 + len(lines) * 10.0 + 10
    box(CX[3], y_, COLW, bh, "#F4F7F8", ec=RULE, lw=1.2)
    ax.add_patch(Rectangle((CX[3], y_), 5, bh, fc=GREEN, ec="none", zorder=2))
    T(CX[3] + 13, y_ + 6, ttl, fit(ttl, COLW - 24, 22, "B"), "B", GREEN_D,
      zorder=4)
    for i, ln in enumerate(lines):
        T(CX[3] + 13, y_ + 28 + i * 10.0, ln, 19, "M", INK, zorder=4)
    y_ += bh + 20

# Lifted 12 mm: at y_ += 16 this card ran to y 948 and crossed the footer
# rule at 944. Now 872..936 — 8 mm of clearance.
y_ += 4
box(CX[3], y_, COLW, 64, GREEN_L, ec=GREEN, lw=1.6)
T(CX[3] + 13, y_ + 8, "Open source · MIT license", 23, "B", GREEN_D, zorder=4)
T(CX[3] + 13, y_ + 30, "Coding sheet for all 80 papers and", 18, "M", INK,
  zorder=4)
T(CX[3] + 13, y_ + 45, "per-paper cells are public.", 18, "M", INK, zorder=4)

# =================================================================
# FOOTER  (944–1030 mm)
# =================================================================
rule(L, 944, CW, lw=1.2, color=RULE)
T(L, 952, "Selected references", 18, "B", INK_SOFT)
# Halved footer: two references at 15 pt and a single right-hand line.
# The licence string is verified against the repository LICENSE (MIT,
# Copyright (c) 2026 Jaewook Kim) before each build of this file.
REFS = [
    "Harvey, Liu & Zhu (2016). …and the Cross-Section of Expected Returns. RFS 29(1).",
    "McLean & Pontiff (2016). Does Academic Research Destroy Return Predictability? JF 71(1).",
]
for i, r_ in enumerate(REFS):
    T(L, 964 + i * 10, r_, fit(r_, 470, 15, "R"), "R", INK_SOFT)

T(R, 952, "Jaewook Kim   ·   jwim1101@gmail.com   ·   orcajae.github.io/valid-framework",
  18, "M", INK_SOFT, ha="right")
T(R, 970, "Non-archival workshop paper · KDD-MLF 2026 · Paper 18 · "
  "ssrn.com/abstract=6508779 · MIT licensed", 16, "SB", INK_SOFT, ha="right")

# =================================================================
OUT = os.path.dirname(os.path.abspath(__file__))
suffix = "_bleed3mm" if BLEED else ""
pdf = os.path.join(OUT, f"kddmlf2026_valid_poster_900x1050{suffix}.pdf")
fig.savefig(pdf, format="pdf", facecolor=PAPER)
print("wrote", pdf)
print(f"  canvas {W + 2 * BLEED:.0f} x {H + 2 * BLEED:.0f} mm"
      f"  (trim {W:.0f} x {H:.0f}, bleed {BLEED:.0f} mm)")

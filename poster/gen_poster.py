#!/usr/bin/env python3
"""
KDD-MLF 2026 · Paper 18 — VALID poster generator.
900 x 1600 mm portrait, vector PDF (text embedded as TrueType).

All numbers traced to:
  CR  = camera-ready  ~/jwquant/paper/kdd-mlf/camera_ready/main.tex
  EXT = extended ver. valid-framework/paper/latex/main.tex (SSRN 6508779)
See GATE0.md for the line-by-line provenance log.
"""
import os
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
if not FACES:                                    # fallback
    for w in ["R", "M", "SB", "B", "EB", "BK", "I", "MI"]:
        FACES[w] = FontProperties(family="Helvetica")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 6

# ----------------------------------------------------------------- palette
INK      = "#13181D"
INK_SOFT = "#4A555F"
RULE     = "#C9D1D6"
GREEN    = "#0E5C42"          # single accent
GREEN_D  = "#083A2A"
GREEN_L  = "#E4F0EA"
RED      = "#B02418"          # failure numbers only
RED_L    = "#FBEAE7"
PAPER    = "#FFFFFF"
SLATE    = "#1E2A32"

W, H = 900.0, 1600.0          # mm
L, R = 45.0, 855.0            # content margins
CW = R - L                    # 810

# 3-column grid (Tier 1 centre stage)
C1X, C1W = 45.0, 210.0
C2X, C2W = 270.0, 360.0
C3X, C3W = 645.0, 210.0
# 3-column grid (Tier 2 evidence band)
E1X, E2X, E3X, EW = 45.0, 320.0, 595.0, 260.0

URL_REPO = "https://github.com/orcajae/valid-framework"
URL_SSRN = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6508779"

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)                     # y grows downward
ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc=PAPER, ec="none", zorder=-10))


# ----------------------------------------------------------------- helpers
def tw(s, size, w="R"):
    """rendered width of s in mm at `size` pt"""
    if not s.strip():
        return 0.0
    return TextPath((0, 0), s, size=size, prop=FACES[w]).get_extents().width / 72 * 25.4


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


def para(x, y, s, max_mm, size=30, w="R", color=INK, lead=1.30, ha="left", zorder=5):
    """returns y after the block (top-origin mm)"""
    lh = size / 72 * 25.4 * lead
    for i, ln in enumerate(wrap(s, max_mm, size, w)):
        T(x, y + i * lh, ln, size, w, color, ha=ha, zorder=zorder)
    return y + len(wrap(s, max_mm, size, w)) * lh


def fit(s, max_mm, size, w="R"):
    """largest size <= `size` that fits max_mm"""
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
    ax.add_patch(Rectangle((x, y), size_mm, size_mm, fc="white", ec="none", zorder=z))
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            if v:
                ax.add_patch(Rectangle((x + (j + quiet) * u, y + (i + quiet) * u),
                                       u * 1.03, u * 1.03, fc=fg, ec="none",
                                       lw=0, zorder=z + 1))


def cax(x, y, w_, h_):
    """chart axes at top-origin mm rect"""
    a = fig.add_axes([x / W, 1 - (y + h_) / H, w_ / W, h_ / H])
    for s in a.spines.values():
        s.set_color(RULE)
        s.set_linewidth(1.0)
    a.tick_params(colors=INK_SOFT, labelsize=24, length=4, width=1.0)
    for lb in a.get_xticklabels() + a.get_yticklabels():
        lb.set_fontproperties(FACES["M"])
    return a


def checkbox(x, y, s=9.0, lw=1.6, color=INK_SOFT, z=6):
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=color, lw=lw, zorder=z))


# =================================================================
# TIER 1 — HEADER  (0–195 mm)
# =================================================================
ax.add_patch(Rectangle((0, 0), W, 14, fc=GREEN, ec="none", zorder=3))

t1 = "VALID: A 12-Item Validation Checklist"
t2 = "for Financial Machine Learning"
TS = min(fit(t1, CW, 118, "EB"), fit(t2, CW, 118, "EB"))
T(L, 44, t1, TS, "EB", INK)
T(L, 44 + TS / 72 * 25.4 * 1.06, t2, TS, "EB", INK)

byline = ("Jaewook Kim  ·  Independent Researcher, JW Quant Research  ·  "
          "KDD-MLF 2026, Jeju  ·  Paper 18 (Oral)")
T(L, 146, byline, fit(byline, 690, 33, "M"), "M", INK_SOFT)

# header QRs (small — the load-bearing QR lives in "The Gate")
qr(755, 126, 46, URL_SSRN)
T(778, 176, "SSRN 6508779", 18, "SB", INK_SOFT, ha="center")
qr(809, 126, 46, URL_REPO)
T(832, 176, "GitHub repo", 18, "SB", INK_SOFT, ha="center")

# =================================================================
# TIER 1 — HOOK BANNER  (196–344 mm)
# =================================================================
ax.add_patch(Rectangle((0, 196), W, 148, fc=GREEN, ec="none", zorder=1))
h1, h2 = "Statistically perfect.", "Economically worthless."
HS = min(fit(h1, CW - 40, 122, "BK"), fit(h2, CW - 40, 122, "BK"))
T(W / 2, 214, h1, HS, "BK", "#8FCFB4", ha="center", zorder=4)
T(W / 2, 214 + HS / 72 * 25.4 * 1.05, h2, HS, "BK", "#FFFFFF", ha="center", zorder=4)
sub = ("340 strategy variants  ·  80 papers audited  ·  "
       "one 12-item gate that tells them apart in ~5 minutes")
T(W / 2, 312, sub, fit(sub, CW - 60, 37, "M"), "M", "#D6EFE3", ha="center", zorder=4)

# =================================================================
# TIER 1 — CENTRE STAGE  (352–650 mm)
# =================================================================
STY = 356          # column heading baseline (top)
for x_, w_, ttl in [(C1X, C1W, "The Field Today"),
                    (C2X, C2W, "The Strategies That Passed Everything"),
                    (C3X, C3W, "The Gate")]:
    T(x_, STY, ttl, fit(ttl, w_, 50, "B"), "B", GREEN)
    rule(x_, STY + 24, w_, lw=2.0, color=GREEN)

# ---------------------------------------------- centre: two contrast cards
CY, CH = 392, 184
cw_ = 172
# PASSED card
box(C2X, CY, cw_, CH, GREEN_L, ec=GREEN, lw=2.0)
ax.add_patch(Rectangle((C2X, CY), cw_, 34, fc=GREEN, ec="none", zorder=2))
T(C2X + 12, CY + 6, "PASSED", 30, "BK", "#FFFFFF", zorder=4)
T(C2X + cw_ - 12, CY + 8, "9 of 340 variants", 21, "SB", "#BFE3D2", ha="right", zorder=4)
for i, (v, k) in enumerate([("Bonferroni: 9", "Holm 9 · BH-FDR 10"),
                            ("t up to 21.7", "Harvey t > 3.0: 10 survive"),
                            ("SR up to 1.98", "vs benchmark 0.917")]):
    yy = CY + 48 + i * 40
    T(C2X + 12, yy, v, 40, "EB", GREEN_D, zorder=4)
    T(C2X + 12, yy + 17, k, 21, "M", INK_SOFT, zorder=4)
T(C2X + 12, CY + CH - 24, "Every classical correction: PASS", 26, "MI", GREEN_D, zorder=4)

# REALITY card
rx = C2X + cw_ + 16
box(rx, CY, cw_, CH, RED_L, ec=RED, lw=2.0)
ax.add_patch(Rectangle((rx, CY), cw_, 34, fc=RED, ec="none", zorder=2))
T(rx + 12, CY + 6, "REALITY", 30, "BK", "#FFFFFF", zorder=4)
T(rx + cw_ - 12, CY + 8, "the same nine", 21, "SB", "#F2C6C0", ha="right", zorder=4)
for i, (v, k) in enumerate([("PBO = 1.0", "all nine — complete overfitting"),
                            ("AUC 0.47–0.64", "near random"),
                            ("DSR: 0 of 340", "E[max SR] under null 2.93")]):
    yy = CY + 48 + i * 40
    T(rx + 12, yy, v, 40, "EB", RED, zorder=4)
    T(rx + 12, yy + 17, k, 21, "M", INK_SOFT, zorder=4)
T(rx + 12, CY + CH - 24, "Corrections cannot see overfitting", 26, "MI", RED, zorder=4)

T(C2X, CY + CH + 8,
  "Multiple-testing control is necessary — and demonstrably not sufficient.",
  25, "MI", INK_SOFT, zorder=4)

# ---------------------------------------------- centre: dark evidence strip
SX, SH = 598, 52
box(C2X, SX, C2W, SH, SLATE, r=5, z=2)
T(C2X + 16, SX + 8,
  "Across all 340 variants: 52% lose money after costs · only 4.4% beat the benchmark.",
  fit("Across all 340 variants: 52% lose money after costs · only 4.4% beat the benchmark.",
      C2W - 32, 27, "SB"), "SB", "#FFFFFF", zorder=4)
T(C2X + 16, SX + 29,
  "All nine trade daily. CPCV rates every one fully overfit; the DSR rejects all 340.",
  fit("All nine trade daily. CPCV rates every one fully overfit; the DSR rejects all 340.",
      C2W - 32, 27, "SB"), "SB", "#9FD8BF", zorder=4)

# ---------------------------------------------- left: the field today
T(C1X, 392, "2.5", 108, "BK", RED)
T(C1X + tw("2.5", 108, "BK") + 10, 415, "of 12", 36, "B", INK)
T(C1X + tw("2.5", 108, "BK") + 10, 439, "items met", 28, "M", INK_SOFT)
T(C1X, 466, "median audited paper (n = 75 empirical, 2018–2026)", 22, "M", INK_SOFT)
T(C1X, 490, "Share of papers failing each dimension", 23, "SB", INK_SOFT)

BAR0, BARW = 148.0, 74.0
rows = [("No code", 85), ("No class balance", 72), ("Costs omitted", 53),
        ("No net performance", 53), ("BnH-only baseline", 33),
        ("Random split", 7), ("CPCV used", 0)]
for i, (lab, val) in enumerate(rows):
    yy = 506 + i * 18
    T(BAR0 - 6, yy - 1, lab, fit(lab, 96, 24, "M"), "M", INK, ha="right")
    ax.add_patch(Rectangle((BAR0, yy), BARW, 11, fc="#EDF1F3", ec="none", zorder=2))
    if val > 0:
        ax.add_patch(Rectangle((BAR0, yy), BARW * val / 100, 11,
                               fc=GREEN if val < 60 else GREEN_D, ec="none", zorder=3))
    T(C1X + C1W, yy - 3, f"{val}%", 26, "EB",
      RED if val == 0 else INK, ha="right")
para(C1X, 628, "This is the reporting norm, not a few bad papers.", C1W, 27, "MI", INK)

# ---------------------------------------------- right: the gate
GY = 392
box(C3X, GY, C3W, 37, GREEN_L, ec=GREEN, lw=1.6)
T(C3X + 10, GY + 4, "Stage 1 — Statistical", 27, "B", GREEN_D, zorder=4)
T(C3X + 10, GY + 21, "V1–V6 · balance, temporal split, CPCV+PBO, variance, permutation",
  15, "M", INK_SOFT, zorder=4)
ax.add_patch(FancyArrow(C3X + C3W / 2, GY + 40, 0, 9, width=2.4, head_width=9,
                        head_length=5, fc=GREEN, ec="none", zorder=4,
                        length_includes_head=True))
box(C3X, GY + 53, C3W, 37, GREEN_L, ec=GREEN, lw=1.6)
T(C3X + 10, GY + 57, "Stage 2 — Economic", 27, "B", GREEN_D, zorder=4)
T(C3X + 10, GY + 74, "V7–V12 · net costs, cost sweep, baselines, regimes, turnover, code",
  15, "M", INK_SOFT, zorder=4)
T(C3X, GY + 94, "Binary items · fixed order · pass/fail", 22, "MI", INK_SOFT)

MY = 500
T(C3X, MY, "False positives on signal-free data (n = 200)", 22, "SB", INK)
for i, (lab, val, ci, col) in enumerate([
        ("AUC alone", "27%", "[21, 34]", RED),
        ("+ CPCV & PBO", "0%", "[0, 1.9]", GREEN)]):
    yy = MY + 20 + i * 24
    T(C3X, yy, lab, 26, "M", INK)
    T(C3X + C3W - 46, yy - 5, val, 34, "EB", col, ha="right")
    T(C3X + C3W, yy + 1, ci, 20, "M", INK_SOFT, ha="right")
T(C3X, MY + 70, "One item (V4) does it — used by 0 of 75.", 23, "SB", GREEN_D)

# big QR — chest/eye height (1,050–1,160 mm above floor)
QY = 592
box(C3X, QY, C3W, 106, "#F1F6F3", ec=GREEN, lw=2.4)
qr(C3X + 9, QY + 9, 88, URL_REPO)
tx = C3X + 105
T(tx, QY + 10, "Run the", 30, "B", GREEN_D, zorder=4)
T(tx, QY + 29, "checklist", 30, "B", GREEN_D, zorder=4)
T(tx, QY + 48, "on your", 30, "B", GREEN_D, zorder=4)
T(tx, QY + 67, "own paper", 30, "B", GREEN_D, zorder=4)
T(tx, QY + 88, "~5 minutes", 26, "BK", RED, zorder=4)
T(C3X, QY + 110, "github.com/orcajae/valid-framework", 20, "M", INK_SOFT, zorder=4)

# =================================================================
# TIER 2 — EVIDENCE BAND  (662–1180 mm)
# =================================================================
rule(L, 722, CW, lw=2.0, color=RULE)
EY = 738
for x_, ttl, sub_ in [(E1X, "Bull Bias", "structural class imbalance (V1–V2)"),
                      (E2X, "Cost Illusion", "net-of-cost performance by frequency (V7–V8, V11)"),
                      (E3X, "Who Is This For?", "adoption paths")]:
    T(x_, EY, ttl, fit(ttl, EW, 54, "B"), "B", INK)
    T(x_, EY + 24, sub_, 23, "M", INK_SOFT)

# ---------------------------------------------- 1. bull bias
a = cax(E1X + 6, 794, EW - 12, 158)
labels = ["BTC\nCatBoost", "ETH\nCatBoost", "SOL\nCatBoost", "BTC\nLSTM"]
unb = [97.2, 90.5, 97.0, 57.7]
bal = [42.3, 45.2, 32.2, 52.3]
xs = range(len(labels))
a.bar([i - 0.20 for i in xs], unb, width=0.38, color=RED, label="unbalanced")
a.bar([i + 0.20 for i in xs], bal, width=0.38, color=GREEN, label="balanced")
a.axhline(50, color=INK_SOFT, ls=(0, (4, 3)), lw=1.4)
a.text(3.55, 52, "50%", fontproperties=FACES["M"], fontsize=20, color=INK_SOFT, ha="right")
a.set_xticks(list(xs))
a.set_xticklabels(labels)
a.set_ylim(0, 105)
a.set_yticks([0, 50, 100])
a.set_yticklabels(["0", "50", "100"])
a.set_ylabel('predicted "long"  (%)', fontproperties=FACES["M"], fontsize=21, color=INK_SOFT)
a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)
lg = a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              prop=FACES["M"], fontsize=21, handlelength=1.2, handletextpad=0.5,
              columnspacing=1.4)
for t_ in lg.get_texts():
    t_.set_color(INK_SOFT)
    t_.set_fontsize(21)

y_ = 976
y_ = para(E1X, y_, "Tree models predict “long” 90–97% of the time. After balancing, "
                   "AUC converges to ≈0.50 — a coin flip.", EW, 30, "SB", INK)
y_ += 12
for lab, val, col in [("Balanced AUC (BTC / ETH / SOL)", "0.50 · 0.49 · 0.51", INK),
                      ("Net SR, 1h balanced", "−0.88 · −1.28 · −0.21", RED),
                      ("18 combinations, BTC mean long", "85%", INK),
                      ("BTC mean balanced AUC / net SR", "0.484 / −0.174", RED)]:
    T(E1X, y_, lab, 24, "M", INK_SOFT)
    T(E1X + EW, y_ - 1, val, 25, "B", col, ha="right")
    rule(E1X, y_ + 16, EW, lw=0.8)
    y_ += 24
T(E1X, y_ + 4, "All 1h models: PBO = 1.000.", 25, "SB", RED)

# ---------------------------------------------- 2. cost illusion
b = cax(E2X + 6, 794, EW - 12, 158)
TFD = [("15m", -1.798, 9, 100), ("1h", -1.323, 133, 96),
       ("4h", 0.276, 9, 44), ("1d", 0.541, 77, 22)]
xs = range(len(TFD))
b.bar(list(xs), [d[1] for d in TFD], width=0.62,
      color=[RED if d[1] < 0 else GREEN for d in TFD])
for i, d in enumerate(TFD):
    va_ = "top" if d[1] < 0 else "bottom"
    off = -0.09 if d[1] < 0 else 0.09
    b.text(i, d[1] + off, f"{d[1]:+.2f}", ha="center", va=va_,
           fontproperties=FACES["B"], fontsize=23,
           color=RED if d[1] < 0 else GREEN)
b.axhline(0, color=INK, lw=1.4)
b.set_xticks(list(xs))
b.set_xticklabels([f"{d[0]}\nn={d[2]}" for d in TFD])
b.set_ylim(-2.35, 1.05)
b.set_yticks([-2, -1, 0, 1])
b.set_ylabel("median net Sharpe, 18 bp", fontproperties=FACES["M"],
             fontsize=21, color=INK_SOFT)
b.spines["top"].set_visible(False)
b.spines["right"].set_visible(False)

y_ = 976
y_ = para(E2X, y_, "Net Sharpe falls monotonically with trading frequency. At 15 minutes "
                   "every single variant loses money after 18 bp costs.", EW, 30, "SB", INK)
y_ += 12
for lab, fr in [("", 0), ("median", 0.62), ("best", 0.82), ("< 0", 1.0)]:
    T(E2X + EW * fr, y_, lab, 22, "SB", INK_SOFT, ha="left" if fr == 0 else "right")
y_ += 22
for tf, med, best, neg in [("15m  (n = 9)", "−1.80", "−1.33", "100%"),
                           ("1h  (n = 133)", "−1.32", "+0.32", "96%"),
                           ("4h  (n = 9)", "+0.28", "+0.57", "44%"),
                           ("1d  (n = 77)", "+0.54", "+1.98", "22%")]:
    T(E2X, y_, tf, 24, "M", INK)
    for val, fr in [(med, 0.62), (best, 0.82), (neg, 1.0)]:
        T(E2X + EW * fr, y_ - 1, val, 25, "B",
          RED if val.startswith("−") or val == "100%" else INK, ha="right")
    rule(E2X, y_ + 15, EW, lw=0.8)
    y_ += 22
T(E2X, y_ + 2, "One 1h CatBoost run: gross SR 0.750 → net 0.135 at 18 bp", 24, "SB", RED)
T(E2X, y_ + 19, "— 82% of gross Sharpe consumed at 101 trades / yr.", 24, "SB", RED)
T(E2X, y_ + 40, "340-variant corpus, 18 bp round-trip, model variants only.", 20, "M", INK_SOFT)

# ---------------------------------------------- 3. who is this for
cards = [("Reviewers & PCs",
          "Require the 12 items as a minimum reporting standard for financial ML "
          "submissions — the role TRIPOD plays in clinical prediction."),
         ("Validation & model-risk teams",
          "A two-stage gate for incoming strategies: statistical items first, "
          "economic items second, both binary."),
         ("Allocators & operational due diligence",
          "Twelve questions to ask before trusting a pitched Sharpe ratio.")]
y_ = 794
for ttl, body in cards:
    lines = wrap(body, EW - 30, 28, "M")
    bh = 34 + len(lines) * 12.6 + 14
    box(E3X, y_, EW, bh, "#F4F7F8", ec=RULE, lw=1.2)
    ax.add_patch(Rectangle((E3X, y_), 6, bh, fc=GREEN, ec="none", zorder=2))
    T(E3X + 18, y_ + 10, ttl, fit(ttl, EW - 32, 32, "B"), "B", GREEN_D, zorder=4)
    for i, ln in enumerate(lines):
        T(E3X + 18, y_ + 36 + i * 12.6, ln, 28, "M", INK, zorder=4)
    y_ += bh + 16

y_ += 10
box(E3X, y_, EW, 68, GREEN_L, ec=GREEN, lw=1.6)
T(E3X + 16, y_ + 11, "Open source · MIT license", 30, "B", GREEN_D, zorder=4)
T(E3X + 16, y_ + 34, "Coding sheet for all 80 papers and", 25, "M", INK, zorder=4)
T(E3X + 16, y_ + 49, "per-paper cells are public.", 25, "M", INK, zorder=4)

# =================================================================
# TIER 2 — INTERACTION STRIP  (1196–1320 mm)
# =================================================================
rule(L, 1196, CW, lw=2.0, color=RULE)
T(L, 1212, "Score the last paper you reviewed", 44, "B", INK)
cbx = L + tw("Score the last paper you reviewed", 44, "B") + 16
for i in range(12):
    checkbox(cbx + i * 13, 1216, s=11)
T(cbx + 12 * 13 + 8, 1220, "median in our audit: 2.5 / 12", 28, "SB", RED)

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
COLX = [L, L + 295]
for i, (tag, name) in enumerate(ITEMS):
    col, row = i // 6, i % 6
    x_ = COLX[col]
    y_ = 1252 + row * 16
    checkbox(x_, y_ - 1, s=11)
    T(x_ + 17, y_ - 3, tag, 27, "EB", GREEN)
    T(x_ + 52, y_ - 3, name, 27, "M", INK)
T(L, 1252 + 6 * 16 + 6, "V1–V6  Stage 1 (statistical)          V7–V12  Stage 2 (economic)",
  23, "SB", INK_SOFT)

HX = 634
box(HX, 1240, R - HX, 114, "#FFFFFF", ec=RED, lw=2.4)
T(HX + 16, 1252, "Self-audit of this paper", 30, "B", RED, zorder=4)
T(HX + 16, 1274, "9 / 12", 56, "BK", INK, zorder=4)
T(HX + 16 + tw("9 / 12", 56, "BK") + 10, 1292, "fully satisfied", 24, "M", INK_SOFT, zorder=4)
T(HX + 16, 1308, "Three partials disclosed: V5 (flatness criteria", 23, "M", INK, zorder=4)
T(HX + 16, 1322, "disagree), V8 (single 18 bp cost level),", 23, "M", INK, zorder=4)
T(HX + 16, 1336, "V10 (3 crypto assets only).", 23, "M", INK, zorder=4)

T(L, 1372, "v1.0 — not a certification. The checklist has had no formal Delphi review; "
           "community revision is invited.", 26, "MI", INK_SOFT)

# =================================================================
# TIER 3 — FOOTER  (1390–1600 mm) — references / contact only
# =================================================================
rule(L, 1390, CW, lw=1.2, color=RULE)
T(L, 1404, "Selected references", 24, "B", INK_SOFT)
REFS = [
    "Harvey, Liu & Zhu (2016). …and the Cross-Section of Expected Returns. RFS 29(1).",
    "McLean & Pontiff (2016). Does Academic Research Destroy Return Predictability? JF 71(1).",
    "Hou, Xue & Zhang (2020). Replicating Anomalies. RFS 33(5).",
    "Bailey & López de Prado (2014). The Deflated Sharpe Ratio. J. Portfolio Management 40(5).",
    "Witzany (2021). A Bayesian Approach to Measurement of Backtest Overfitting. Risks 9(1).",
    "Kapoor & Narayanan (2023). Leakage and the Reproducibility Crisis in ML-based Science. Patterns 4(9).",
]
for i, r_ in enumerate(REFS):
    col, row = i // 3, i % 3
    T(L + col * 410, 1428 + row * 16, r_, fit(r_, 400, 22, "R"), "R", INK_SOFT)

rule(L, 1490, CW, lw=1.2, color=RULE)
T(L, 1504, "Non-archival workshop paper · Full version: SSRN Working Paper 6508779",
  24, "SB", INK_SOFT)
T(L, 1526, "jwim1101@gmail.com   ·   @jwquant   ·   github.com/orcajae/valid-framework",
  24, "M", INK_SOFT)
T(R, 1504, "KDD-MLF 2026 · 9th ACM SIGKDD Workshop", 24, "SB", INK_SOFT, ha="right")
T(R, 1526, "on Machine Learning in Finance · Jeju, Korea", 24, "M", INK_SOFT, ha="right")
T(R, 1556, "© 2026 Jaewook Kim · Code MIT-licensed", 22, "R", INK_SOFT, ha="right")

# =================================================================
OUT = os.path.dirname(os.path.abspath(__file__))
pdf = os.path.join(OUT, "kddmlf2026_valid_poster_900x1600.pdf")
fig.savefig(pdf, format="pdf", facecolor=PAPER)
print("wrote", pdf)

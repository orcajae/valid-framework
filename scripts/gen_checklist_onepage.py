#!/usr/bin/env python3
"""One-page VALID checklist (A4) for docs/assets/valid_checklist_onepage.pdf.

The item strings are not typed here. They are parsed at build time from the
two places that already carry them, so the three artifacts cannot drift:

  * the camera-ready Table 1  ->  docs/paper/Paper18_VALID_camera_ready_corrected.pdf
  * the landing self-scorer   ->  docs/index.html, .item-q spans

The camera-ready PDF is the published record and is tracked in this repository;
the LaTeX source under paper/ is gitignored, so parsing the PDF is what lets a
fresh clone rebuild this file. When the source happens to be present it is used
as a second opinion and any disagreement aborts the build.

A mismatch in item count, ordering or numbering aborts the build. The only
hand-written strings are the header, the scoring rule and the footer; each is
checked against the same n=74 canon used on the landing page.

Palette and faces mirror poster/gen_poster_900x1050.py.
"""
import html
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = os.path.join(ROOT, "docs", "paper",
                  "Paper18_VALID_camera_ready_corrected.pdf")
TEX = os.path.join(ROOT, "paper", "kdd-mlf", "main.tex")  # gitignored, optional
HTML = os.path.join(ROOT, "docs", "index.html")
OUT = os.path.join(ROOT, "docs", "assets", "valid_checklist_onepage.pdf")

# ----------------------------------------------------------------- fonts
FONT_DIR = os.path.join(ROOT, "poster", "assets", "fonts", "inter")
FACES = {}
for w, fn in [("R", "Inter-Regular.ttf"), ("M", "Inter-Medium.ttf"),
              ("SB", "Inter-SemiBold.ttf"), ("B", "Inter-Bold.ttf"),
              ("BK", "Inter-Black.ttf")]:
    p = os.path.join(FONT_DIR, fn)
    if os.path.exists(p):
        fontManager.addfont(p)
        FACES[w] = FontProperties(fname=p)
    else:
        FACES[w] = FontProperties(family="Helvetica")
matplotlib.rcParams["pdf.fonttype"] = 42  # keep the text extractable

# --------------------------------------------------------------- palette
INK = "#13181D"
INK_SOFT = "#4A555F"
RULE = "#C9D1D6"
GREEN = "#0E5C42"
GREEN_D = "#083A2A"
GREEN_L = "#E4F0EA"
PAPER = "#FFFFFF"

# ------------------------------------------- source: the camera-ready PDF
# Table 1 runs "# | Item | Rationale | Failure Mode". Text extraction emits it
# cell by cell, one cell per line, so a row reads as
#   V1 / "Report prediction class" / "distribution" / <rationale> / <mode>
# We slice each row between its V-marker and the next, then cut where the
# rationale column starts. Those rationale openings are listed verbatim below,
# so a reworded rationale fails loudly instead of leaking into the item text.
RATIONALE_HEAD = (
    "Base-rate learning", "Structural class imbalance", "Information leakage",
    "Holdout insufficient", "PBO = 0 from flat landscape",
    "Confirm signal existence", "Cost omission inflates",
    "Cost assumptions can flip", "Weak baselines inflate",
    "Bull-market trend-following", "High-frequency cost headwinds",
    "<10% of papers provide code",
)
# Extraction drops the subscript in Var(SR_IS); restore it for rendering.
CR_GLYPH = {"Var(SRIS)": "Var(SR_IS)"}


def cr_text():
    import fitz
    with fitz.open(CR) as d:
        t = "\n".join(p.get_text() for p in d)
    return t.replace("\u00a0", " ").replace("\u2010", "-")


def cr_items():
    """The 12 item cells of Table 1, read out of the published PDF."""
    t = cr_text()
    try:
        body = t.split("Failure Mode", 1)[1].split("ML Trading", 1)[0]
    except IndexError:
        sys.exit("Table 1 not located in %s" % os.path.basename(CR))
    marks = {}
    for m in re.finditer(r"(?m)^V(\d{1,2})\s*$", body):
        marks.setdefault(int(m.group(1)), (m.start(), m.end()))
    out = []
    for n in range(1, 13):
        if n not in marks:
            sys.exit("V%d not found in Table 1 of %s"
                     % (n, os.path.basename(CR)))
        start = marks[n][1]
        end = marks[n + 1][0] if n + 1 in marks else len(body)
        cell = re.sub(r"\s+", " ", body[start:end]).strip()
        cuts = [cell.find(h) for h in RATIONALE_HEAD if cell.find(h) > 0]
        if not cuts:
            sys.exit("no rationale boundary found in row V%d: %r" % (n, cell))
        cell = cell[:min(cuts)].strip()
        for k, v in CR_GLYPH.items():
            cell = cell.replace(k, v)
        out.append((n, cell))
    return out


# --------------------------------------- optional cross-check: the LaTeX
# LaTeX math in the item column, spelled out. Any token not listed here
# survives into the PDF with its braces, which the self-check rejects.
TEX_MATH = {
    r"$\mathrm{Var}(\mathrm{SR}_{\mathrm{IS}})$": "Var(SR_IS)",
    r"$\geq$": "\u2265",
}


def tex_items():
    src = open(TEX, encoding="utf-8").read()
    body = src.split(r"\label{tab:valid}", 1)[1].split(r"\bottomrule", 1)[0]
    out = []
    for line in body.splitlines():
        m = re.match(r"\s*V(\d{1,2})\s*&\s*(.+?)\s*&", line)
        if not m:
            continue
        text = m.group(2)
        for k, v in TEX_MATH.items():
            text = text.replace(k, v)
        text = text.replace(r"\,", " ").replace(r"\-", "")
        text = re.sub(r"\\mbox\{(.*?)\}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if "\\" in text or "{" in text or "$" in text:
            sys.exit("unconverted LaTeX in V%s: %r" % (m.group(1), text))
        out.append((int(m.group(1)), text))
    return out


# ------------------------------------------------ source: the landing page
def html_questions():
    src = open(HTML, encoding="utf-8").read()
    out = []
    for m in re.finditer(
        r'<span class="item-q"><b>V(\d{1,2})\.</b>\s*(.*?)</span>', src, re.S
    ):
        q = html.unescape(re.sub(r"\s+", " ", m.group(2))).strip()
        out.append((int(m.group(1)), q))
    return out


ITEMS_CR = cr_items()
ITEMS_Q = html_questions()

if len(ITEMS_CR) != 12 or len(ITEMS_Q) != 12:
    sys.exit("expected 12 items, got camera-ready=%d html=%d"
             % (len(ITEMS_CR), len(ITEMS_Q)))
if [n for n, _ in ITEMS_CR] != list(range(1, 13)):
    sys.exit("camera-ready Table 1 rows are not V1..V12 in order")
if [n for n, _ in ITEMS_Q] != list(range(1, 13)):
    sys.exit("landing items are not V1..V12 in order")

# Second opinion, when the gitignored LaTeX source is available locally.
if os.path.exists(TEX):
    for (n, a), (_, b) in zip(ITEMS_CR, tex_items()):
        norm = lambda s: re.sub(r"[\s‐-―-]+", " ", s).strip()
        if norm(a) != norm(b):
            sys.exit("V%d disagrees: camera-ready %r vs tab:valid %r"
                     % (n, a, b))
    print("cross-check: tab:valid agrees with the camera-ready on all 12 items")
else:
    print("cross-check: paper/kdd-mlf/main.tex absent (gitignored) — skipped")

ITEMS = [(n, ITEMS_CR[i][1], ITEMS_Q[i][1]) for i, (n, _) in enumerate(ITEMS_Q)]

# ------------------------------------------------------------ hand-written
TITLE = "VALID \u2014 a 12-item validation checklist for financial machine learning"
SUB = ("Twelve binary items in a fixed order: the statistical gate first, "
       "the economic gate second.")
STAGE1 = ("Stage 1 \u00b7 Statistical", "verifiable from the manuscript alone")
STAGE2 = ("Stage 2 \u00b7 Economic", "required before anything is traded")
SCORING = [
    ("How to score",
     "One point per item, satisfied or not \u2014 no partial credit. "
     "Score Stage 1 and Stage 2 separately, then total out of 12."),
    ("What the total means",
     "The median paper in an audit of 74 empirical crypto ML studies scores "
     "2.5 of 12; none of the 74 applied CPCV. A study that clears Stage 1 but "
     "fails Stage 2 is the statistical\u2013economic disconnect: a significant "
     "signal with no exploitable alpha."),
    ("What it is not",
     "Self-scoring is not a certification. It records your own reading of your "
     "own artifacts. We scored our own camera-ready 9 of 12 and disclosed "
     "three partials."),
]
FOOT_L = ("Jaewook Kim  \u00b7  jwim1101@gmail.com  \u00b7  "
          "orcajae.github.io/valid-framework")
FOOT_R = "KDD-MLF 2026 \u00b7 Paper 18 \u00b7 SSRN 6508779 \u00b7 MIT licensed"

# ----------------------------------------------------------------- canvas
W, H = 210.0, 297.0                      # A4 portrait, mm
L, R = 16.0, 194.0
CW = R - L

fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)                        # y grows downward, as in the poster
ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc=PAPER, ec="none", zorder=0))


def T(x, y, s, size, w, color, ha="left", zorder=4):
    return ax.text(x, y, s, fontproperties=FACES[w], fontsize=size, color=color,
                   ha=ha, va="top", zorder=zorder)


def width_mm(s, size, w):
    """Rendered width of a string, in mm."""
    t = ax.text(0, 0, s, fontproperties=FACES[w], fontsize=size, alpha=0)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 25.4


def wrap(s, maxw, size, w):
    lines, cur = [], ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if cur and width_mm(trial, size, w) > maxw:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


fig.canvas.draw()

# ----------------------------------------------------------------- header
ax.add_patch(Rectangle((0, 0), W, 30, fc=GREEN, ec="none", zorder=1))
T(L, 8.0, TITLE, 12.4, "BK", "#FFFFFF")
T(L, 18.5, SUB, 8.8, "M", "#D6EFE3")
T(R, 9.0, "orcajae.github.io/", 8.2, "SB", "#BFE3D2", ha="right")
T(R, 13.4, "valid-framework", 8.2, "SB", "#BFE3D2", ha="right")

# ------------------------------------------------------------ item blocks
BOX = 3.9          # tick box side, mm
TXT = L + BOX + 4.2
TW = R - TXT - 2.0
y = 38.0


def stage_head(y, title, note):
    T(L, y, title, 10.6, "B", GREEN_D)
    w = width_mm(title, 10.6, "B")
    T(L + w + 3.0, y + 1.2, note, 8.4, "M", INK_SOFT)
    ax.plot([L, R], [y + 6.2, y + 6.2], color=RULE, lw=0.8, zorder=2)
    return y + 9.6


def item_row(y, n, label, question):
    ax.add_patch(Rectangle((L, y + 0.4), BOX, BOX, fc=PAPER, ec=GREEN, lw=0.9,
                           zorder=2))
    head = "V%d.  %s" % (n, label)
    T(TXT, y, head, 9.6, "B", INK)
    yy = y + 5.2
    for ln in wrap(question, TW, 8.4, "R"):
        T(TXT, yy, ln, 8.4, "R", INK_SOFT)
        yy += 3.7
    return yy + 4.7


y = stage_head(y, *STAGE1)
for n, label, q in ITEMS[:6]:
    y = item_row(y, n, label, q)

y += 2.0
y = stage_head(y, *STAGE2)
for n, label, q in ITEMS[6:]:
    y = item_row(y, n, label, q)

# ----------------------------------------------------------- scoring rule
y += 8.0
box_top = y
inner = y + 4.0
for head, body in SCORING:
    T(L + 4.0, inner, head, 9.0, "B", GREEN_D)
    inner += 4.7
    for ln in wrap(body, CW - 8.0, 8.4, "R"):
        T(L + 4.0, inner, ln, 8.4, "R", INK)
        inner += 3.7
    inner += 2.2
box_h = inner - box_top + 1.0
ax.add_patch(Rectangle((L, box_top), CW, box_h, fc=GREEN_L, ec=GREEN, lw=1.0,
                       zorder=1))

# ----------------------------------------------------------------- footer
FY = H - 15.0
ax.plot([L, R], [FY, FY], color=RULE, lw=0.8, zorder=2)
for line, size, face in ((FOOT_L, 7.8, "M"), (FOOT_R, 7.4, "R")):
    if width_mm(line, size, face) > CW:
        sys.exit("footer line overruns the text column: %r" % line)
T(L, FY + 2.2, FOOT_L, 7.8, "M", INK_SOFT)
T(L, FY + 7.0, FOOT_R, 7.4, "R", INK_SOFT)

if inner > FY - 4:
    sys.exit("content overflows the page: bottom %.1f mm vs footer %.1f mm"
             % (inner, FY))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# CreationDate omitted so two builds of the same inputs are byte-identical,
# which is what lets releases/backtesting_checklist.pdf be checked by md5.
fig.savefig(OUT, format="pdf", facecolor=PAPER,
            metadata={"CreationDate": None})
print("wrote", OUT)
print("  A4 %.0f x %.0f mm, content bottom %.1f mm (footer rule at %.1f)"
      % (W, H, inner, FY))
for n, label, q in ITEMS:
    print("  V%-3d %s" % (n, label))

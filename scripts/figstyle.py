"""
Shared figure style for the manuscript figures.

Every generator in this directory imports this module so that the six figures
read as one set. The palette and typeface are those of the camera-ready
workflow diagram (`paper/kdd-mlf/figures/gen_workflow.py`): Times serif on a
monochrome blue hierarchy, with a single warm accent reserved for reference
lines and a neutral gray for the "before" condition in before/after pairs.

Type size is kept honest by authoring each figure at the width it will occupy
on the page. The manuscript text block is 5.45 in wide, so a figure included at
`width=\\textwidth` must be authored at `TEXT_W` and one at `width=0.8\\textwidth`
at `0.8 * TEXT_W`. LaTeX then scales by 1.0 and a 9 pt label in the generator
is a 9 pt label in the PDF. Figures authored larger and scaled down were the
reason the earlier set had labels ranging from roughly 4 pt to 10 pt.

Use `width(fraction)` to size a figure and `apply()` before creating it.
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg switch)

# Text block width of the elsarticle `preprint,12pt` class, in inches.
TEXT_W = 5.45

# --- palette ------------------------------------------------------------
# Blue hierarchy, light to dark.
FILL_LIGHT = "#E6F1FB"
FILL_MID = "#B5D4F4"
FILL_STRONG = "#378ADD"
STROKE = "#185FA5"
INK = "#0C447C"
INK_DEEP = "#042C53"

# Neutral gray: the unbalanced or "before" condition, never a category of
# its own.
GRAY_FILL = "#B4B2A9"
GRAY_STROKE = "#5F5E5A"

# Warm accent, reserved for reference lines, thresholds and medians so that
# a reader can tell a datum from a rule at a glance.
ACCENT = "#C7402F"

# Categorical states, used only where a cell must encode pass/partial/fail.
STATE_PASS = "#3E8E5A"
STATE_PARTIAL = "#E8B33C"
STATE_FAIL = "#C7402F"
STATE_NA = "#C9C9C4"

RULE = "#333333"
GRIDLINE = "#DDDDDD"


def width(fraction=1.0):
    """Figure width in inches for an ``\\includegraphics[width=f\\textwidth]``."""
    return TEXT_W * fraction


def apply(base=9.0):
    """Install the shared rcParams. Call once, before creating a figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 0.5,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.titlepad": 6.0,
        "axes.labelpad": 3.5,
    })


def despine(ax, keep=("left", "bottom")):
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


def panel(ax, text):
    """Left-aligned panel title, e.g. ``(a) Directional prediction share``."""
    ax.set_title(text, loc="left", color=INK_DEEP)


def hgrid(ax):
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out):
    """Write the vector figure and a PNG companion beside it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    return out

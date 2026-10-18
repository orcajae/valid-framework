# Derivation scripts

Each script recomputes a number or figure that appears in the papers. None of
them trains a model or runs a backtest: they read stored artifacts and print or
plot them, so any published value can be traced back to a file.

> **October 2026.** Reproducing a stored value does not make it valid. Several
> stored values were computed under a different definition from the one the
> paper printed. Items marked *withdrawn* below are withdrawn in
> [CHANGELOG_CR.md, E13](../CHANGELOG_CR.md#e13-most-empirical-results-withdrawn-after-a-full-re-verification);
> see [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).

## Runs from a fresh clone

These need nothing beyond `results/reference/` in this repository:

| Script | Reproduces |
|---|---|
| `derive_corpus_stats.py` | 52% negative (178/340), which stands; 4.4% above the benchmark (15/340), *withdrawn* because the count depends on the annualisation scale |
| `derive_tf_ladder.py` | net Sharpe by trading frequency at 18 bp. The ordering and the share of negative variants stand; the medians are *withdrawn* (√252 applied to per-event returns at every frequency) |
| `gen_fig0_valid_overview.py` | framework overview diagram (no data; layout only) |
| `gen_fig1_bullbias_340.py` | bull-bias figure (prediction share and post-balancing AUC) |
| `gen_fig2_340.py` | net Sharpe distribution figure (workshop paper) |
| `gen_fig6_340.py` | net Sharpe distribution figure (extended version) |

## Needs an artifact outside this repository

Two experiment outputs are not tracked here. Point the scripts at them with an
environment variable, or they fall back to the author's local layout:

| Script | Required input | Override |
|---|---|---|
| `derive_bonferroni.py` | `multiple_testing.json` (*withdrawn*: its t-statistics mix annualised Sharpe ratios with event counts, and its DSR benchmark is unscaled) | `MT_JSON=/path/to/multiple_testing.json` |
| `gen_fig5_regime_real.py` | `regime_conditional.json` (two-state HMM regime split) | `REGIME_JSON=/path/to/regime_conditional.json` |

Both files are available on request and are included in the SSRN supplementary
material.

## Output paths

Figure generators write next to the manuscript sources by default, which is
outside this repository. Set the output explicitly when running from a clone:

```sh
FIG1_OUT=./fig1.pdf python3 scripts/gen_fig1_bullbias_340.py
FIG2_OUT=./fig2.pdf python3 scripts/gen_fig2_340.py
FIG5_OUT=./fig5.pdf REGIME_JSON=./regime_conditional.json python3 scripts/gen_fig5_regime_real.py
FIG6_OUT=./fig6.pdf python3 scripts/gen_fig6_340.py
FIG0_OUT=./fig0.pdf python3 scripts/gen_fig0_valid_overview.py
```

## Shared style

`figstyle.py` holds the palette, typeface and type sizes that every generator
imports, so the figure set reads as one system: Times serif on a monochrome
blue hierarchy, a warm accent reserved for reference lines and thresholds, and
neutral gray for the "before" condition of a before/after pair. It also embeds
TrueType fonts (`pdf.fonttype = 42`), so no output contains Type 3 fonts.

Each figure is authored at the width it occupies on the page --- `fs.width(1.0)`
for `width=\textwidth`, `fs.width(0.78)` for `width=0.78\columnwidth` --- so
LaTeX scales by 1.0 and a 9 pt label in a generator is 9 pt in the PDF. Figures
drawn at some other size and scaled were why the earlier set ranged from about
4 pt to 11 pt across figures.

See `CHANGELOG_CR.md` for what each number corrects.

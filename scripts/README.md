# Derivation scripts

Each script recomputes a number or figure that appears in the papers. None of
them trains a model or runs a backtest: they read stored artifacts and print or
plot them, so any published value can be traced back to a file.

## Runs from a fresh clone

These need nothing beyond `results/reference/` in this repository:

| Script | Reproduces |
|---|---|
| `derive_corpus_stats.py` | 52% negative (178/340), 4.4% above the benchmark (15/340), and the same counts excluding the 18 cost-level entries |
| `derive_tf_ladder.py` | net Sharpe by trading frequency at 18 bp (15m −1.80, 1h −1.32, 4h +0.28, 1d +0.54) |
| `gen_fig1_bullbias_340.py` | bull-bias figure (prediction share and post-balancing AUC) |
| `gen_fig2_340.py` | net Sharpe distribution figure |

## Needs an artifact outside this repository

Two experiment outputs are not tracked here. Point the scripts at them with an
environment variable, or they fall back to the author's local layout:

| Script | Required input | Override |
|---|---|---|
| `derive_bonferroni.py` | `multiple_testing.json` (five corrections, nine Bonferroni survivors, DSR) | `MT_JSON=/path/to/multiple_testing.json` |
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
```

Figure generators embed TrueType fonts (`pdf.fonttype = 42`), so the resulting
PDFs contain no Type 3 fonts.

See `CHANGELOG_CR.md` for what each number corrects.

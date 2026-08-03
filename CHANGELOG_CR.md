# Changelog — camera-ready revision (2026-08-01)

Baseline: `main_v1_submitted.tex.bak` (submitted camera-ready, 2026-07-07).
Evidence: `valid-framework/poster/RUN_ATTRIBUTION_AUDIT.md`.
No experiment was re-run for this revision; every changed figure is read from a
stored artifact, and each change carries a `% src:` comment in the source.

## E1 — Table 5, 15-minute row: cost level and trade count mislabelled

The row reported a Sharpe ratio measured at 50 bp inside a table captioned at
18 bp; at 18 bp the value is +0.186. The trade-count entry ("500+") was the
total number of trades over the evaluation window (579, ≈74 per year), not
trades per year. The row has been removed with the rest of the table (E2).
Source: `results/reference/variants_340.csv`
(`COST_ML_15m_18bp = 0.1858`, `COST_ML_15m_50bp = -0.222`).

## E2 — Table 5 replaced: the ML rows were not model output

The "1h ML" and "15m ML" series of the cost table, and the "1h ML balanced" row
of the baselines file, were produced by random signals
(`src/paper_kbs_phase1.py`, lines 424–439: `np.random.seed(42)`, `ml_sig`,
`ml_15m_sig`) applied to daily BTC returns. No model prediction entered those
numbers, and the paper labelled them "ML (CatBoost, bal.)". The affected values
are 0.640, 0.530, 0.416, 0.186, 0.335 and −0.222, together with the derived
"55–91% alpha erosion" and the "−64% / −91%" cost-drag entries.

Table 5 is replaced by a frequency ladder computed from the 340-variant corpus
(`scripts/derive_tf_ladder.py`): median net Sharpe at 18 bp of −1.80 (15m, n=9,
all negative), −1.32 (1h, n=133), +0.28 (4h, n=9), +0.54 (daily, n=77). The
cost claim that survives measurement is stated in Section 4.3: in the one run
where gross and net Sharpe are both recorded, costs consume 82% of the gross
Sharpe ratio (0.750 → 0.135) at 101 trades per year
(`results/round_a_orderflow/backtest_with_costs.csv`).
The paper's real BTC 1h CatBoost (balanced) result, net SR −0.88, was already
reported in Table 4 and is unchanged.

## E3 — Section 4.4 rewritten: metrics came from two different runs

The submitted text attributed PBO = 0.000, permutation p = 0.000, AUC = 0.570
and net SR = +0.135 to a single strategy. The AUC, p-value and net SR come from
the order-flow 1h run, whose own PBO is 0.267 and whose stored verdict records
`PBO (all): 0.267 — FAIL`. The PBO of 0.000 comes from a separate 1h CPCV run
(AUC 0.547) that produced neither a permutation test nor a net-of-cost
backtest. No run in the archive satisfies PBO ≈ 0, permutation p < 0.05 and a
materially sub-benchmark net Sharpe at the same time.

The section now uses the order-flow run as reported: permutation passes, CPCV
fails at PBO = 0.267, and the corpus-level version of the argument (nine
Bonferroni survivors, all PBO = 1.0, none surviving the Deflated Sharpe Ratio)
carries the claim.

## E4 — V5 flatness verdict withdrawn

Var(SR_IS) = 0.012 and the IS–OOS correlation are outputs of
`src/paper_extras.py`, which runs the 1h balanced CPCV sweep and takes the
variance of in-sample Sharpe across model configurations within each fold. The
statistic therefore belongs to the ML pipeline, and both figures stand. Its own
recorded interpretation, however, is "NOT FLAT: Var(SR_IS) = 0.0123 > 0.01",
which contradicts the submitted "flat parameter landscape" verdict. The verdict
is withdrawn: the paper now states that the absolute criterion (0.01) and the
Monte Carlo null criterion (95th percentile 0.328) disagree, which is itself the
argument for pre-specifying the null under V5. "First empirical confirmation of
Witzany (2021)" becomes "empirical evidence consistent with" that analysis, in
both Section 2 and Section 4.4.

## E5 — Appendix F: IS–OOS correlation corrected

"Across the 15 CPCV folds, fold-level ρ = −0.08 (p = 0.60)" did not match the
stored result. `paper/kdd-mlf/results/is_oos_correlation.json` reports a
configuration-level CPCV correlation of ρ = −0.833 (p = 0.005, n = 9
configurations). The appendix and Section 4.4 now report ρ = −0.83 and state
that both reported correlations are configuration-level. The walk-forward
figure (ρ = −0.73, n = 26) matched the artifact and is unchanged.

## E6 — Monte Carlo setting labels

The four Monte Carlo settings were all generated from daily-bar bootstrap
series: every stored iteration carries 838–919 events per series, which is
daily-scale sampling (`results/paper_kbs/mc_fpr_all_settings_200.csv`). The
label mismatch for two of the settings was already recorded in the repository's
`REPRODUCE.md`; a footnote now extends that record to all four columns of
Table 7. The false-positive results are unaffected — every setting yields 0%
under CPCV with PBO.

## Self-assessment revised downward

V8 (cost sensitivity) moves from satisfied to partial: all corpus variants are
evaluated at a single 18 bp level, so the paper reports a frequency ladder
rather than a cost sweep of its own strategies. The overall self-assessment
becomes 9/12 fully satisfied and 3/12 partial (V5, V8, V10), from 10/12 and
2/12. Evidence pointers for V9 and V11 now cite the experimental setup and
Section 4.3 instead of the replaced table.

## Figures

Figure 2 was checked for the same class of problem as E2 and is clean. Its
generator read a file named `all_482_variants.csv` (540 rows) and dropped the
200 Monte-Carlo null rows, leaving exactly the 340-variant corpus; the
histogram bin counts, the 52% negative share and the 4.4% beat share are
identical to those computed from `results/reference/variants_340.csv`. The 15
identifiers that differ between the two files are label-only renames with
identical values. The figure is therefore a correct rendering of the corpus and
its content is unchanged.

The extended version's distribution figure is named `fig6_482_distribution` for
historical reasons only; like Figure 2 it is rendered from the 340-variant
corpus, and the filename is legacy.

Both included figures were re-rendered with TrueType font embedding
(`pdf.fonttype = 42`) so that the PDF contains no Type 3 fonts; a 300 dpi pixel
comparison confirms that only glyph rasterisation differs, with all bars, axes,
boxes and rules identical. Figure 2 is now generated by
`scripts/gen_fig2_340.py`, which reads the corpus file directly. The v1 figure
files are kept in `figures/_v1_fonttype3_backup/`.

## Corpus composition

The 340-variant corpus contains 18 cost-level entries: the rule-based DM+CB
benchmark and two signal proxies, each evaluated at 0, 5, 10, 18, 30 and 50 bp.
Twelve of the eighteen are the proxies withdrawn under E2; the six benchmark
entries are real rule-based backtests. A footnote at the first mention of the
corpus now records what the headline figures look like without them: excluding
the 18 entries leaves 322 variants, of which 55.0% are negative net of costs and
11 exceed the benchmark. The printed corpus-level figures (52%, 178/340
negative; 4.4%, 15/340 above the benchmark) are retained as the figures for the
corpus as constructed.

Two properties of the counts are worth stating plainly. First, five of the
fifteen entries above the benchmark are benchmark entries themselves — the
benchmark evaluated at 0, 5, 10 and 18 bp plus its row in the baseline block —
so ten model variants exceed 0.917 on the rounded threshold. Section 4.5 and
the Figure 2 caption now state this split explicitly ("15 variants (4.4%) …
of which 10 are model variants"); the abstract retains the unqualified 4.4%.
All ten model variants carry PBO = 1.0. Second, none of
the nine Bonferroni survivors is a cost-level or baseline entry: every survivor
is a daily-frequency model variant with net Sharpe at or above 1.08, while the
highest cost-level entry reaches 0.954. The multiple-testing conclusion is
therefore unaffected by the composition of these rows.
Recomputation: `scripts/derive_corpus_stats.py`, `scripts/derive_bonferroni.py`.

## E8 — Figures rebuilt or withdrawn

Four figures of the extended version were audited against their generators.

**Figure 1 (bull bias)** is regenerated from the 340-variant corpus. Its four
panels previously used hard-coded prediction shares — 91/9, 60.4/39.6 and 93/7
(`src/paper_figures.py` L48, L60, L102) — none of which match the paper's own
bull-bias table (97.2/42.3, 90.5/45.2, 97.0/32.2, 57.7/52.3). The new figure
reads those values from `results/reference/variants_340.csv` and matches the
table exactly. Generator: `scripts/gen_fig1_bullbias_340.py`.

**Figure 1, panel (c)** is removed. It plotted bear-market equity curves for
"unbalanced" and "balanced" models whose signals were drawn by
`np.random.choice` (`src/paper_figures.py` L69–98); no model output entered it.

**Figure 4 (complexity vs performance)** is removed. Its eight points were
hard-coded feature-count/Sharpe pairs (`src/paper_figures.py` L229–238) for a
strategy family outside the 340-variant corpus, with no artifact behind them.
No prose claim depended on the figure; the caption carried the only claim.

**Figure 5 (regime-conditional performance)** is regenerated from
`paper/kdd-mlf/results/regime_conditional.json` and loses its third series. The
DM and buy-and-hold bars matched the artifact (calm 0.6883 / 0.5058, stressed
1.4008 / 0.7923, 35.1% of days) and are retained; the "Best ML (1h balanced)"
bars were hard-coded (`ml_srs = [0.30, 0.50]  # approximate`,
`src/paper_figures.py` L292). No regime-conditional Sharpe ratio was ever
computed for any model variant, so the series is dropped rather than
re-estimated, and the caption states this. Generator:
`scripts/gen_fig5_regime_real.py`.

Both new figures embed TrueType fonts (`pdf.fonttype = 42`). The four withdrawn
images are moved out of the repository to `~/jwquant/_v31_figures_backup/`.

## E9 — Figure 1 (VALID compliance heatmap) rebuilt

The heatmap shipped as a raster image dated 2026-04-10 with no generator in the
repository. Two defects were found.

**Layout.** The Pass/Partial/Fail/N/A legend was drawn inside the heatmap axes
and landed on top of the column labels, which were themselves clipped at the
figure edge — the reader saw "V1: Class Dist." and "V2: Balance..." with the
legend text crossing them. The legend now occupies a row of its own below the
figure, the column labels sit above the heatmap where nothing competes for the
space, and the median is keyed beside the failure-rate panel.

**Item V9.** Its failure-rate bar read 0%, next to V10 and V11 at 0%. V10 and
V11 are 0% because every paper is coded N/A; V9 is coded for all 75 papers and
24 of them fail it. The corrected bar reads 32%.

The figure is now generated from `results/reference/literature_audit_80.csv` by
`scripts/gen_fig7_valid_heatmap.py`, which maps the seven coding dimensions onto
the twelve items using the correspondence already printed in Table 3. The
reported median of 2.5 of 12 reproduces exactly. Item failure rates use
applicable-only denominators and the caption now says so, which is why V1 and V2
read 95% and 98% here against the 72% of Table 3: that table divides by all 75
papers, the figure by the 59 where a directional class balance applies. V5 and
V6 were never coded per paper — the coding sheet has no column for either — and
the caption now records them as asserted universal non-compliance rather than
letting the all-red columns imply per-paper evidence. The output is vector PDF
with TrueType fonts embedded, replacing the raster PNG.

## E10 — The figure set rebuilt as one system

All six figures now come from a generator in `scripts/`, share `figstyle.py`,
and ship as vector PDFs. Two of them had no generator at all before this pass,
and three were rasters. The manuscript PDF drops from 1.88 MB to 0.85 MB with
zero Type 3 fonts and nothing unembedded.

**Type size.** Each figure is now authored at the width it occupies on the page,
so LaTeX scales it by 1.0. The old set was drawn at arbitrary sizes and scaled:
Figure 5 was drawn 3.4 in wide and enlarged by 1.25, Figure 1's row labels were
shrunk to about 3.8 pt, and the same nominal 8 pt label came out anywhere from
4 pt to 11 pt depending on the figure.

**Figure 2 (framework overview)** is rebuilt. Its gate boxes had been laid out
with a gap smaller than their own rounded-corner padding, so every internal
border doubled; the "Key Insight" band overlapped both the Stage 1 container
and the Bull Bias box; connectors curved from the gates across the containers
to the failure-mode boxes; and the title was clipped off the canvas. The
connectors are dropped rather than redrawn — each failure-mode box already
names its detecting items. Generator: `scripts/gen_fig0_valid_overview.py`.

**Figure 4 (PBO paradox)** is rebuilt from `results/reference/variants_340.csv`
and `monte_carlo_fpr_200.csv`. Panel (a) had been a scatter of AUC against PBO
whose y axis carries no variance: all 209 variants with a PBO value have PBO of
exactly 1.000. The caption's "nearly all configurations receive PBO close to
1.0" is corrected to state the result, and the panel now draws the 0.20
threshold that V4 accepts below, so the gap to it is the visible quantity. Panel
(b) gains the absolute flatness threshold of 0.01, the second of the two
criteria the text reports as disagreeing; the null 95th percentile recomputes to
0.3278 against the printed 0.328. Generator: `scripts/gen_fig2_pbo_paradox.py`.

**Collisions removed.** Figure 1: the legend band sat against the V1–V12 labels
above it, and "V12: Code available" ran into the score panel's axis label, which
moved to the foot of that panel. Figure 3: the panel (a) legend sat on the first
bar and the "50%" and "0.50 (chance)" captions ran into the reference lines and
the bars — both values are already axis ticks, so the two panels share one
legend beneath the figure and the reference line is keyed there. Figure 4: the
panel (b) legend sat on the tallest bin and the rotated AUC label crossed the
V4 line. Figure 6: the filled callout badges, the only boxed annotations in the
set, are flattened.

**Equations.** All six numbered equations render, and three symbols that were
used before being introduced are now defined: $\tau$ (elapsed time since the
event, Equations 1–3), $\hat{\sigma}$ (the local volatility estimate the
barriers scale to), and $M$ (the number of candidate configurations, used in
Equations 4 and 5). The mean $\overline{\mathrm{SR}}_{\mathrm{IS}}$ in Equation 5
is now named as well.

## E11 — The empirical set corrected from 75 papers to 74

**How it surfaced.** Rebuilding the compliance heatmap from the coding sheet
(E10) put the per-paper V4 column on the page for the first time. One cell came
back green: id 9, Arian, Mobarekeh and Seco (2024), coded
`D4_validation = CPCV/Bagged CPCV`. The audit table on the facing page prints
`CPCV used 0/75`, and the body prints "no empirical paper in our sample uses
it". The figure and the table it was drawn beside disagreed.

**The cause.** The exclusion rule lived in code rather than in the coding guide,
and it tested one column:

```python
empirical = df[~df["method"].str.contains("Survey|Synthetic|PBO|Anomaly|microstructure|DL survey", case=False, na=False)]
```

Id 9's `method` reads `CPCV variants`, which matches nothing in that pattern, so
the row stayed in the empirical set. The same row carries the pattern's own
keywords in three other columns: `assets = Synthetic`, `best_metric = PBO
comparison`, `D1_cost = N/A synthetic`. The paper is a synthetic-controlled
methodological study — its title is "Backtest overfitting in the machine
learning era: A comparison of out-of-sample testing methods in a synthetic
controlled environment", and our own camera-ready introduces it that way in
Section 2. It was never a crypto ML trading study. The other synthetic paper in
the sheet, Witzany (2021), was excluded only because its `method` string
happened to read `PBO analysis`; the difference in treatment was an artifact of
which column the regular expression could see, not a difference in kind.

**The fix.** The inclusion criterion is now stated in `audit/coding_guide.md`
rather than implied by a regular expression:

> A paper is 'empirical' iff it applies an ML method to real cryptocurrency
> market data for a trading or prediction task. Excluded: surveys,
> synthetic-only methodological studies, non-crypto asset studies, non-ML
> studies. Excluded ids: 1, 9, 11, 12, 71, 79.

`audit/audit_analysis.py` selects on that id list; the superseded regular
expression is retained as a comment. Both manuscripts now print the criterion,
which neither did before. **The coding sheet itself is unchanged — all 80 rows
stand as coded.** Only the classification rule moved.

**Every affected figure, in full.** All values are transcribed from
`audit/audit_analysis.py` and `scripts/gen_fig7_valid_heatmap.py` stdout; none
was recomputed by hand.

| Quantity | Was (n = 75) | Now (n = 74) |
|---|---|---|
| Empirical papers | 75 | 74 |
| D1: Costs omitted | 40/75, 53% [42%, 64%] | 40/74, 54% [43%, 65%] |
| D2: No class balance | 54/75, 72% [61%, 81%] | 54/74, 73% [62%, 82%] |
| D3: Random split | 5/75, 7% [3%, 15%] | 5/74, 7% [3%, 15%] |
| D4: Weak validation | 15/75, 20% [13%, 30%] | 15/74, 20% [13%, 31%] |
| D4: CPCV used | 0/75, 0%, no CI printed | 0/74, 0% [0%, 5%] |
| D5: BnH-only baseline | 25/75, 33% [24%, 44%] | 26/74, 35% [25%, 46%] |
| D6: No net performance | 40/75, 53% [42%, 64%] | 40/74, 54% [43%, 65%] |
| D7: No code available | 64/75, 85% [76%, 92%] | 63/74, 85% [75%, 91%] |
| Median VALID score | 2.5 / 12 | 2.5 / 12 |
| Figure 1: V1, V2 denominator | 59 applicable papers | 58 applicable papers |
| Figure 1: V1, V2 fail rate | 95%, 98% of 59 | 95%, 98% of 58 |

**The headline results do not move.** CPCV adoption is **0 of 74** — the claim
the figure appeared to contradict survives the correction that the figure
prompted. The median audited paper still satisfies **2.5 of 12** VALID items
(`scripts/gen_fig7_valid_heatmap.py`: `papers=74, median score=2.50/12`). Class
imbalance remains the most neglected dimension and reproducibility the second.
The excluded paper failed none of the seven dimensions, so the correction moves
the denominators, not the numerators.

**D5 corrected.** The previously printed 25/75 (33%) understated the script's
count by one; the rebuilt pipeline now prints the script value directly: 26/74
(35%). D7 moves for the same reason in the opposite direction — its count is now
63 where 64 was printed — and its rate, which the script had computed as 84%
against a printed 85%, now agrees at 85%.

**A misdescription corrected.** The camera-ready said "Of the 80 papers, 75 are
empirical and 5 are surveys". Only two of the six excluded entries are surveys;
the composition is two surveys, two synthetic-only methodological studies, one
equities-only anomaly replication and one non-ML microstructure study. Both
manuscripts now itemize the exclusions by kind instead of calling them all
surveys.

**Surfaces updated**: SSRN §3.1 and Table 3, camera-ready §4.1 and Table 3,
Figure 1 and its caption, `results/reference/audit_summary_80.csv`,
`poster/gen_poster.py` (both the audit bar chart and the two n captions),
`docs/index.html`, `docs/scorer.js`.

Three residual prose instances of the pre-correction percentages (72%, 53%,
33%) were aligned with Table 3 in the hosted camera-ready on Aug 4.

## Unchanged

The literature audit (Table 3), bull bias (Table 4), the ablation and Monte
Carlo false-positive results (Tables 6–7), the multiple-testing analysis and
Bonferroni survivor table (Table 8, nine survivors, t from 4.07 to 21.70, all
PBO = 1.0, zero surviving the Deflated Sharpe Ratio, E[max SR] = 2.93), and the
corpus-level figures of 52% negative net Sharpe (178/340) and 4.4% exceeding the
benchmark (15/340) were each recomputed from their stored artifacts and match
the printed values. Scripts: `scripts/derive_corpus_stats.py`,
`scripts/derive_bonferroni.py`, `scripts/derive_tf_ladder.py`.

---

These corrections were identified by applying the VALID audit process to our
own camera-ready before the workshop. Conclusions are unchanged; the corrected
version is the one presented at KDD-MLF 2026.

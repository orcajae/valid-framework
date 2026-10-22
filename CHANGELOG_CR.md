# Changelog — camera-ready revision (2026-08-01)

> **October 2026: most empirical results of the paper are withdrawn. See [E13](#e13-most-empirical-results-withdrawn-after-a-full-re-verification). The pre-registered re-measurement of PBO and the permutation test is [E14](#e14-pbo-and-the-permutation-test-re-measured-under-a-pre-registered-design). Its pre-registered robustness study, which sets the scope of two E14 conclusions, is [E15](#e15-the-e14-findings-re-tested-under-a-pre-registered-robustness-design). The corpus figures that E13 kept are restated in [E16](#e16-the-340-variant-corpus-re-counted-e13s-corpus-figures-restated).** Entries below are in the order they were made.

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

> E7 was allocated to a suspected Figure 2 corpus contamination and withdrawn
> when the check came back clean; see "Figures" below. No E7 correction exists.
> [numbering annotated 2026-08-07; content unchanged]

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

## E12: Priority claims withdrawn after a concurrent standard

**What changed.** The camera-ready (abstract, Section 1 and Section 6) and the
SSRN version described VALID as, to our knowledge, the first checklist-style
protocol for financial ML to pair binary statistical and economic gates in a
fixed reporting order. This README went further and called VALID "the first
domain-specific checklist for financial ML" without that qualifier, and the
camera-ready abstract stated that no domain-specific validation standard exists
for financial ML.

**How it surfaced.** On 14 October 2026 we read Khushi (2026), "QFRS:
Quantitative Finance Reporting Standards for Forecasting, Evaluation and
Trading Claims" (Artificial Intelligence Review,
doi:10.1007/s10462-026-11664-w). The journal received QFRS on 29 March 2026,
before VALID first appeared on SSRN in April 2026. QFRS defines seven
reviewer-enforceable standards for financial forecasting and trading claims,
from dataset specification to cost-aware backtesting, and its audit of 41 forex
forecasting papers published in 2025 found none compliant with all seven. The
two frameworks were developed independently.

**The correction.** A priority claim does not survive a concurrent framework
with the same structure, so both statements are withdrawn: in this README and
the extended manuscript now, and in the next SSRN revision. The camera-ready
presented at KDD-MLF 2026 stays byte-identical
(md5 f2a901b81576f59dc114e7322734ddcc) and is superseded on both points by this
entry.

**What we claim instead.** QFRS and VALID standardize different layers. QFRS
specifies how an experiment is built and reported, including a cost-aware
backtest, the number of trades and class prevalence. VALID adds gates that
account for the number of configurations tried (CPCV with PBO), a permutation
test and a comparison against simple rule-based baselines, and it measures how
often its own gates pass strategies with no signal (27–30% for AUC-based
validation alone, 0% with CPCV and PBO). We describe the two as complementary.

**Unchanged.** No number, table or figure changes.

## E13: Most empirical results withdrawn after a full re-verification

**How it surfaced.** On 18 and 19 October 2026, while preparing a journal
version, we re-derived every printed number from the stored outputs, compared
every method description with the code that produced the number, and tried to
locate every paper listed in the literature-audit sheet. Earlier corrections
(E1–E12) had only compared printed numbers with stored outputs. This pass found
three new kinds of failure: the audit sheet does not describe the papers it
lists; several statistics were computed under a different definition from the
one printed; and some statistics were attributed to the wrong run.

**Withdrawn.**

1. *Literature audit* (Section 3, the audit table, the compliance heatmap, the
   Appendix B coding subset, and every statement about what "the literature"
   does). We searched Crossref, OpenAlex and the web for each of the 74 rows
   coded as empirical:
   - 15 rows matched a paper as cited (first author, year and venue). Four of
     those 15 are rule-based momentum studies, not machine learning.
   - 41 rows matched a real paper only with a different author, venue, year or
     method.
   - For 18 rows we found no matching paper.
   - Two pairs of rows describe the same paper under different author names.
   - None of the ten rows printed in Appendix B "to enable independent
     verification" matches the paper it names.

   Withdrawn: 80 surveyed / 74 empirical papers, the median of 2.5 of 12, the
   D1–D7 rates, "no empirical paper uses CPCV", and every code attributed to an
   individual paper. The sheet was produced in March 2026 by AI-assisted web
   search, with the codes entered directly into the file. No record of the
   sources consulted was kept, and the rows were never checked against the
   papers. Publishing it unverified was the author's error. We apologise to the
   authors whose work was misdescribed.

2. *PBO-based results.* The paper defines PBO as the out-of-sample rank of the
   in-sample-best configuration among several (Bailey et al. 2017). The
   experiments instead computed the share of CPCV paths in which one model's
   in-sample accuracy exceeded its out-of-sample accuracy (KNOWN_ISSUES.md #1).
   That share is 1.0 in all 1,027 stored values, real and synthetic. A
   diagnostic with a planted signal (holdout AUC up to 0.69) also gives 1.0, so
   the gate rejects everything.

   Withdrawn: "CPCV with PBO reduces false positives from 27% to 0%", the
   ablation table and "V4 is the single most important item", "every Bonferroni
   survivor carries PBO = 1.0", and Figure 2(a). The only rank-based PBO values
   (0.267 for the order-flow run, 0.000 for a second 1h run) rank three or fewer
   configurations. They are withdrawn as evidence.

3. *Multiple-testing survivors.* The t-statistics multiplied an annualised
   Sharpe ratio by the square root of the number of events in the whole series
   (KNOWN_ISSUES.md #5). Recomputed in consistent units:
   - Bonferroni, Holm, Benjamini–Hochberg and t > 3 each leave 0 survivors.
   - The largest t is 0.76, not 21.70.

   Withdrawn: "nine of 340 variants survive Bonferroni correction with t up to
   21.70", and the "significance without validity" argument built on it.

4. *Parameter-space variance (V5).* The real value of 0.012 is a variance
   across three configurations. The Monte Carlo null distribution is a variance
   across the folds of a single model (KNOWN_ISSUES.md #2). These are different
   quantities, so the comparison is withdrawn: the null percentiles 0.328 and
   0.430, and "the two criteria disagree".

5. *Cost-ladder magnitudes.* Per-event returns were annualised with √252 at
   every bar frequency (KNOWN_ISSUES.md #4).

   Withdrawn: the median net Sharpe ratios by frequency (−1.80, −1.32, +0.28,
   +0.54), "only daily variants beat the benchmark", and "15 of 340 (4.4%)
   exceed the benchmark". These depend on the scale.

6. *Monte Carlo permutation gate.* The gate used 20 shuffles per series, below
   the checklist's own minimum of 100. Its permutation models were trained with
   half the boosting iterations of the real model, and half of the BTC series
   had no permutation test at all (KNOWN_ISSUES.md #3).

   Withdrawn: the 18–22% permutation pass rates, and every combined-gate rate
   that includes them.

7. *Section 4.4 correlations.*
   - ρ = −0.83 came from a different run (daily features, nine configurations)
     from the order-flow 1h run it was attached to.
   - The walk-forward ρ = −0.73 compares one fixed momentum benchmark across
     rolling windows. It involves no configurations and no machine learning, and
     it is −0.08 without a |SR| < 10 filter.

   Both are withdrawn.

8. *Self-assessment.* The score of 9 of 12 rested partly on items 2–6 above
   and is withdrawn.

**What still stands.** In each item below, the definition in the paper, the
computation in the code and the stored value agree.

- 178 of the 340 variants (52%) have a negative net Sharpe ratio at 18 bp. The
  sign does not depend on the annualisation error.
- The share of negative variants falls with bar frequency: 15-minute 9 of 9,
  1-hour 96%, 4-hour 44%, daily 22% (model variants).
- Bull bias: without class weighting, tree models predict "long" for 90–97% of
  test observations (BTC 97.2%, ETH 90.5%, SOL 97.0%). Class weighting brings
  this to 32–45%, with AUC near 0.50. The deep-learning models range from 58% to
  93%; for the two MLP rows the "balanced" column repeats the unbalanced value.
- On 200 signal-free series per asset (i.i.d. bootstrap of daily returns, which
  removes autocorrelation and volatility clustering), AUC > 0.55 flags 27.0%
  (BTC), 30.0% (ETH) and 26.5% (SOL) as positive.
- In the order-flow 1h run:
  - transaction costs consume 82% of the gross Sharpe ratio (0.750 → 0.135) at
    101 trades per year;
  - the run passes a 200-permutation test (AUC 0.570 against a shuffled 95th
    percentile of 0.516, p < 0.005; previously printed as p = 0.000).
- No variant passes the Deflated Sharpe Ratio once the benchmark is computed as
  Bailey and López de Prado (2014) define it. The benchmark is 2.73 in the units
  of the stored Sharpe ratios, against a best observed value of 1.98, and the
  highest DSR is 0.33. The printed comparison used an unscaled 2.93.

**Other errors found in the same pass.**
- The Monte Carlo has three daily-bar assets, not four settings: the "BTC daily"
  setting repeats the BTC inputs.
- Events per series are 820–942 (BTC), 595–725 (ETH) and 498–666 (SOL).
- The deep-learning long bias spans 58–93%, not 58–86%.
- In the SSRN version only:
  - 5 of 24 cross-section momentum configurations are positive, not 4.
  - The multi-asset balanced long shares for BTC and SOL are 51% and 46%.

**What changes in this repository.**
- Removed from the current version (they remain in the git history):
  - the audit coding sheet, its summary and analysis script;
  - the camera-ready PDF, whose Appendix B assigns codes to named papers;
  - the audit heatmap, the PBO figure and their generators.
- Kept as records: the stored outputs in `results/reference/`, which still
  reproduce from their inputs and whose meaning is corrected here, and the
  poster files.
- Updated: the README, the landing page, the scorer and the one-page
  checklist.
- `KNOWN_ISSUES.md` lists the code-level causes.
- `valid.metrics.deflated_sharpe_ratio` now scales its benchmark (0.2.1).
- The twelve checklist items are unchanged as a proposal. What is withdrawn is
  the empirical support the paper claimed for them.

**What happens next.** PBO and the parameter-space variance will be recomputed
as the paper defines them, under a pre-registered design that also measures the
power of each gate on series with a planted signal. A literature audit will
appear again only if it is rebuilt from a logged search, with every code tied to
a quoted passage. Results will be reported in a new entry whatever they show.

**Supersedes.**
- E12's sentence quoting "0% with CPCV and PBO".
- The "Unchanged" section below.
- The KDD-MLF 2026 camera-ready and SSRN v3.2, on every point above.

## E14: PBO and the permutation test re-measured under a pre-registered design

E13 said that PBO and the parameter-space variance would be recomputed as the
paper defines them, under a pre-registered design that also measures the power
of each gate. This entry reports that run, whatever it shows. The revised paper
(SSRN v3.3) gives the full analysis.

- Design: [`experiments/true_pbo/PREREGISTRATION.md`](experiments/true_pbo/PREREGISTRATION.md).
  It was fixed before the new statistics were computed, with amendments A1–A4.
  The AUC rates on the same signal-free series were already known from E13.
- Code: [`experiments/true_pbo/`](experiments/true_pbo/). The run script is the
  executed one with private paths replaced by an environment variable.
- Outputs: [`results/true_pbo/`](results/true_pbo/). Per-series results,
  every configuration's in-sample and out-of-sample Sharpe ratio on every
  split, and every permuted AUC.

**Design.**
- Signal-free series: the E13 Monte Carlo inputs, an i.i.d. bootstrap of daily
  BTC, ETH and SOL returns, 200 series each (same data sha256; AUC > 0.55
  reproduced 27.0%, 30.0% and 26.5%). The bootstrap keeps each asset's mean
  return, so the series carry drift.
- Planted signal: 100 BTC series at each of two strengths. After the bootstrap,
  r′(t+1) = r(t+1) + β·σ·z(t), where z(t) is the standardised five-day return up
  to day t. β = 0.117 and 0.163 were fixed on separate calibration seeds before
  the evaluation seeds were run. Mean holdout AUC: 0.560 and 0.606.
- PBO: the paper's Eq. 1 (rank / (M + 1), logit ≤ 0) over the 15 CPCV splits
  (N = 6, k = 2) of eight CatBoost configurations (depth 4 or 6 × learning rate
  0.03 or 0.1 × with or without class weights). Each configuration is refitted
  on every split, and in-sample performance is its per-event Sharpe ratio on
  the training events it was fitted to. This applies Bailey et al.'s rank
  statistic to refitted CPCV splits; it is not their CSCV procedure.
- Permutation test: 100 shuffles of the whole label vector, each refitted with
  the same model and settings as the real fit.

**Results.** Pass rate, %, with 95% Wilson intervals in `results/true_pbo/gate_rates.csv`.

| Gate | Signal-free BTC | ETH | SOL | Planted β = 0.117 | β = 0.163 |
|---|---|---|---|---|---|
| AUC > 0.55 | 27.0 | 30.0 | 26.5 | 53.0 | 79.0 |
| Permutation | 17.5 | 14.0 | 16.0 | 44.0 | 71.0 |
| PBO < 0.20 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| PBO < 0.50 | 31.5 | 31.5 | 32.5 | 21.0 | 8.0 |
| Net Sharpe > 0 | 43.0 | 48.5 | 49.5 | 69.0 | 89.0 |
| AUC, permutation and net Sharpe | 12.0 | 10.5 | 11.5 | 37.0 | 66.0 |

Every permutation pass is also an AUC pass, so the last row equals permutation
and net Sharpe combined.

**What this means.**
- **PBO, computed this way, is not a test for an edge.**
  - At 0.20 it passes 0–1% of series with or without a signal. With 15 splits,
    passing needs the in-sample best in the upper half out of sample on at least
    13 splits.
  - At 0.50 it passes 31.5% of signal-free BTC series, 21% at the weaker
    planted signal (Fisher p = 0.058, not significant) and 8% at the stronger
    one (p < 0.001).
- **What drives it.**
  - In-sample selection picks one of the two depth-6, learning-rate-0.1
    configurations on 94–99% of splits, with or without a signal.
  - Which configurations do best out of sample depends on the data:
    - on signal-free series, the ones without class weights, which collect the
      retained drift;
    - with the stronger signal, the slower learners.
  - In both cases the in-sample pick is not an out-of-sample leader. PBO
    describes how well in-sample ranks carry over within the grid.
  - These are observations on one grid and one in-sample measure. Canonical
    CSCV and a validation-slice in-sample measure were not run.
- **Guidance for V4:** report PBO with the grid and the in-sample measure, as
  a description of the search, not as a pass-or-fail gate. The item's wording
  is unchanged.
- **The permutation test separates signal from noise** (44% and 71% against
  14.0–17.5%) but runs at about three times its nominal 5%. Two untested
  candidates:
  - overlapping triple-barrier labels, whose serial dependence a full shuffle
    removes;
  - a holdout that is not purged from the training events.
  V6 stands, with that qualification.
- **Real data.** PBO is 0.80 on the actual BTC daily series; 11.2% of
  signal-free series reach 0.80 or more. It is also 0.80 on the hourly pipeline
  of the paper, which differs from the daily null (hourly bars, four-bar
  payoff, embargo), so that comparison is not like for like.
- **V5 remains unresolved.** The hourly Var(SR_IS), recomputed per event across
  the same three configurations as before, is 4.88e-5; multiplied by 252 it
  gives 0.0123, which is the stored 0.012. It lies below all 600 signal-free
  values, but those are daily series, so the comparison is not like for like.
- **Secondary three-configuration grid** (reported, not a gate, as
  pre-registered): PBO < 0.20 passes none of the 800 series; PBO < 0.50 passes
  9.5–14.5% of signal-free and 1–2% of planted-signal series.
- **KNOWN_ISSUES 6** (two PBO thresholds) is settled in the sense that neither
  threshold makes PBO a test for an edge in this pipeline.

**Library change (0.2.2).** `valid.metrics.compute_pbo` used the relative rank
rank / M; Eq. 1 uses rank / (M + 1). The two agree when M is even, as in the
eight-configuration grid and in the worked example, whose output is unchanged.
They differ when M is odd: on the three-configuration grid they disagreed for
almost every series. The function now follows Eq. 1.

**Limits.**
- The null is an i.i.d. bootstrap that keeps drift and removes volatility
  clustering.
- The holdout is not purged. The payoff is one bar.
- The net-Sharpe gate charges costs only on changes of predicted class.
- There is one model family and one planted mechanism, which is close to one of
  the features, with 100 series per strength.
- The pre-registration timestamps are in a private repository.

**Supersedes.** E13's "What happens next" for PBO and the parameter-space
variance.

**Addendum (2026-10-21).**
- **Equation numbers.** "Eq. 1" in this entry is the numbering of the
  camera-ready and SSRN v3.2. In SSRN v3.3 the PBO equation is Eq. 2.
- **`results/true_pbo/null.csv`.** The `arm` column was empty in every row,
  because the summary script read the label "null" as a missing value. It now
  reads "null". No other value changed, and the script now reads labels as text.
- **Public scripts.** `run.py` and `summarize.py` now run without the private
  inputs. That covers every signal-free and planted-signal statistic on the
  eight-configuration grid, and the BTC-daily real arm. The hourly arm and the
  three-configuration grid still need the private inputs.
- **E13 overstated one point.** Its "How it surfaced" says that earlier
  corrections had only compared printed numbers with stored outputs. Most did,
  but some traced a table or a figure to its code (for example E2 and E8). None
  compared the definition of every statistic with its computation.
- **Landing page.** It said PBO "almost never passes a strategy". That holds at
  the 0.20 threshold, not at 0.50, and the page now says so.
- **README.**
  - It said the share of net-negative variants "falls with bar frequency"; it
    falls as the bars lengthen.
  - It described the stage split of SSRN v3.2. v3.3 follows the camera-ready
    split, and the README checklist table now uses v3.3's item wording.
- **Robustness study.** The study of these findings is pre-registered in
  [`experiments/inbox7/PREREGISTRATION.md`](experiments/inbox7/PREREGISTRATION.md)
  (committed before any run).

## E15: The E14 findings re-tested under a pre-registered robustness design

E14's addendum pointed to a robustness study of its two findings, pre-registered
before any run. This entry reports that study, as the pre-registration requires
whatever the results. Nothing in E14 is withdrawn: its numbers stay as published,
and this entry sets the scope of two of its conclusions.

- Design: [`experiments/inbox7/PREREGISTRATION.md`](experiments/inbox7/PREREGISTRATION.md),
  committed in bae70d3 before any series was run. Its sha256 (`7ee7d405…fad2`)
  was checked against this repository before the code was written. Its change
  log is empty: the design was not changed.
- Code: [`experiments/inbox7/`](experiments/inbox7/). It imports the E14
  pipeline (`experiments/true_pbo/run.py`) unchanged. The published scripts
  differ from the executed copy only in file paths and comments. Re-running 18
  series with them (two of each arm in the PBO stage; two each of A-ETH, A-SOL
  and B2 in the permutation stage) reproduced every stored matrix and permuted
  AUC bit for bit, and every stored per-series value other than run times to
  within 1e-16, the
  rounding of the stored CSV files.
- Outputs: [`results/inbox7/`](results/inbox7/). Per-series values, the
  matrices behind every PBO (for S3, the per-block counts, sums and sums of
  squares from which every half-split Sharpe ratio is computed), every permuted
  AUC, every rate with its 95% Wilson interval, and the decision file.

**Design.** Daily Binance data with the same sha256 digests as E14.

| Arm | Series | Construction |
|---|---|---|
| A | BTC, ETH, SOL × 200 (seeds 0–199) | i.i.d. bootstrap of daily returns with the sample mean removed |
| B0, B1, B2 | BTC × 100 each (seeds 1000–1099) | As A, plus E14's planted mechanism at β = 0, 0.117 and 0.163. β was not recalibrated after removing the mean |
| B0′ (B0p in the files) | BTC × 100 (seeds 1000–1099) | E14's base with the mean kept, β = 0: the matched control for E14's planted arms |
| C | BTC × 200 (seeds 0–199) | Stationary bootstrap of demeaned returns, mean block length 20 days |

- PBO, four ways, each on an eight-configuration grid with per-event Sharpe
  ratios before costs:
  - S1, E14's method: refitted CPCV (N = 6, k = 2, 15 splits), in-sample
    measured on the training events each configuration was fitted to.
  - S2: as S1, with in-sample measured on a validation slice (the last 20% of
    each split's training events, after a 20-event gap), which is not fitted.
  - S3, canonical CSCV: out-of-sample per-event returns from purged 6-fold
    cross-validation, cut into 16 blocks; all 12,870 half-splits, no refitting.
  - S4: as S1, with eight LightGBM configurations.
- Permutation test, four ways, 100 refitted permutations each. A series passes
  if its real holdout AUC exceeds the 95th percentile of the 100 permuted AUCs.
  - T1, E14's test: full shuffle, holdout not purged.
  - T2: full shuffle, with the 20 training events before the holdout dropped.
  - T3: as T2, with the labels rolled by a random offset (circular shift),
    which keeps the dependence between neighbouring labels.
  - T4: as T2, with the order of blocks of 20 consecutive labels shuffled.
- Decision rules, fixed in advance (pre-registration §3): size is the pass rate
  in B0 for PBO and in A for the permutation test; power is the pass rate in B2.

**Checks.**
- The pre-registered unit tests passed before the run: CSCV gives PBO = 0 when
  one column dominates every block and 1 for identical columns; a circular shift
  keeps the circular lag-1 autocorrelation of a synthetic label vector (the
  ordinary lag-1 changes by at most 0.0025); a one-block shuffle is the identity; the demeaned
  returns have mean below 1e-12 (largest 2.7e-18).
- S1 and T1 on E14's base reproduced the stored E14 values for seeds 0–4: PBO
  and the permuted 95th percentile identical, AUC within 6e-17.
- The timing check passed: projected 1.7 h and 2.2 h against limits of 7.2 h
  and 8.0 h.
- After the run, every PBO was recomputed from the stored matrices (0
  mismatches in 800 series), and every permutation verdict from the stored
  permuted AUCs (0 mismatches in 3,600).

**Implementation choices the pre-registration left open.** None changes a
definition. They are listed in `experiments/inbox7/run.py`.
- B0 and B0′ are generated with β unset. E14's drift step with β = 0 adds
  exactly zero (unit test: identical arrays).
- C's blocks wrap around the end of the series.
- S2: the validation slice is the last 20% of the split's training events in
  time order, and the 20 events before it are dropped.
- S3: rows are cut into 16 blocks whose sizes differ by at most one. A tie for
  the in-sample best goes to the earlier configuration, as in E14.
- S4: LightGBM settings that the pre-registration does not name are left at
  their defaults.
- A PBO that cannot be computed counts as not passing. None occurred.
- G on the purged holdout fits the base model on the events before the
  20-event gap.
- T3 offsets are drawn from ⌊0.1n⌋ to ⌈0.9n⌉ inclusive. T4 blocks are counted
  from the start of the series. T1 and T2 use the same permuted label vectors.
- When an AUC cannot be computed it is set to 0.5, as in E14. AUC equals 0.5
  exactly in 314 of 360,000 permuted fits and 2 of 1,800 real fits; the code
  does not record which of these came from the fallback. A real AUC of 0.5
  cannot pass, since the smallest permuted 95th percentile is 0.545. Excluding
  the permuted values of 0.5 would change one of 3,600 verdicts (T1, one SOL
  series) and no rate in the decision rules.
- The machine slept during part of the PBO stage. This affected run time, not
  results: the computation is deterministic, and the 500 series run in both
  stages, including three that spanned the sleep, gave identical AUC and net
  Sharpe ratios.

**Results: PBO.** Pass rate, % [95% Wilson]. One-sided Fisher exact test of
B2 > B0. Rule: "has power" if B2 − B0 is at least 20 points and p < 0.05; "no
power" if it is under 10 points and p ≥ 0.05; "inconclusive" otherwise.
Intervals for every cell are in `results/inbox7/stage1_rates.csv`.

| Variant | Threshold | B0 | B1 | B2 | B2 − B0 (pp) | p | Verdict | A-BTC | C |
|---|---|---|---|---|---|---|---|---|---|
| S1, E14 method | 0.20 | 0 [0.0, 3.7] | 0 | 0 [0.0, 3.7] | 0 | 1.0 | no power | 0.5 | 1.5 |
| | 0.50 | 48 [38.5, 57.7] | 17 | 6 [2.8, 12.5] | −42 | 1.0 | no power | 52.0 | 57.5 |
| S2, validation slice | 0.20 | 0 [0.0, 3.7] | 1 | 3 [1.0, 8.5] | +3 | 0.12 | no power | 0.0 | 1.5 |
| | 0.50 | 47 [37.5, 56.7] | 64 | 71 [61.5, 79.0] | +24 | 4.5e-4 | has power | 52.0 | 52.5 |
| S3, canonical CSCV | 0.20 | 13 [7.8, 21.0] | 26 | 46 [36.6, 55.7] | +33 | 2.1e-7 | has power | 15.5 | 14.5 |
| | 0.50 | 54 [44.3, 63.4] | 70 | 84 [75.6, 89.9] | +30 | 3.5e-6 | has power | 54.0 | 54.5 |
| S4, LightGBM | 0.20 | 0 [0.0, 3.7] | 0 | 0 [0.0, 3.7] | 0 | 1.0 | no power | 0.5 | 1.0 |
| | 0.50 | 54 [44.3, 63.4] | 36 | 25 [17.5, 34.3] | −29 | 1.0 | no power | 49.5 | 63.5 |

A-BTC and C beside the three cells with power: S2 at 0.50, 52.0 [45.1, 58.8]
and 52.5 [45.6, 59.3]; S3 at 0.20, 15.5 [11.1, 21.2] and 14.5 [10.3, 20.0]; S3
at 0.50, 54.0 [47.1, 60.8] and 54.5 [47.6, 61.3]. Outside the decision rules:
with A-BTC or C as the size reference, S2 at 0.50 would fall short of the
20-point margin (71 against 52.0 and 52.5); the S3 cells would not.

**Outcome for PBO: a variant with power exists** (precedence rule 1): S2 at
0.50, S3 at 0.20 and at 0.50.

Mean PBO, with a series-level bootstrap 95% interval for B0 and B2 (all arms in
`results/inbox7/stage1_mean_pbo.csv`), and the mean holdout AUC of the base
model:

| | A-BTC | C | B0′ | B0 | B1 | B2 |
|---|---|---|---|---|---|---|
| S1 | 0.509 | 0.483 | 0.606 | 0.510 [0.485, 0.535] | 0.624 | 0.712 [0.687, 0.737] |
| S2 | 0.493 | 0.485 | 0.481 | 0.509 [0.484, 0.535] | 0.461 | 0.409 [0.381, 0.437] |
| S3 | 0.483 | 0.463 | 0.443 | 0.482 [0.435, 0.531] | 0.384 | 0.274 [0.229, 0.320] |
| S4 | 0.499 | 0.456 | 0.475 | 0.486 [0.462, 0.511] | 0.545 | 0.585 [0.558, 0.613] |
| Holdout AUC, not purged | 0.503 | 0.536 | 0.517 | 0.514 | 0.558 | 0.604 |

**Results: permutation test.** Pass rate, %. Size is measured on A, power on
B2. Intervals for every cell are in `results/inbox7/stage2_rates.csv`.

| Variant | A-BTC | A-ETH | A-SOL | A pooled: k of 600, Wilson upper bound | B0 | B1 | B2 | Size controlled | Power kept |
|---|---|---|---|---|---|---|---|---|---|
| T1, full shuffle, not purged (E14) | 15.0 | 15.0 | 15.5 | 91, 18.26 | 25 | 43 | 74 | not in the rule | not in the rule |
| T2, full shuffle, purged | 14.0 | 14.5 | 15.5 | 88, 17.72 | 19 | 44 | 75 | no | yes |
| T3, circular shift, purged | 6.5 | 6.5 | 6.0 | 38, 8.57 | 8 | 26 | 48 | yes | no |
| T4, block shuffle, purged | 8.5 | 7.5 | 7.0 | 46, 10.08 | 11 | 25 | 50 | no | yes |

Rule: size is controlled if the rate is at most 7.5% on each asset and the
pooled upper bound is at most 9%; power is kept if B2 is at least 50%. Outside
the decision rules, B2 exceeds B0 under every variant (one-sided Fisher
p ≤ 1.1e-9).

**Outcome for the permutation test: partial.**
- T3 meets the pre-registered size criterion (pooled 6.3% [4.6, 8.6]), but its
  power is 48%, two series short of 50%.
- T4 keeps power at exactly 50%, but fails both size conditions: 17 of 200
  A-BTC series (8.5%), and a pooled upper bound of 10.08% (46 of 600; at most
  40 would meet the bound).
- The thresholds were fixed in advance and are not adjusted.

**Other pre-registered rates.** Pass rate, %, on the same holdout: not purged
for T1, purged for T2–T4. Every permutation pass also had AUC > 0.55 on that
holdout, so AUC combined with Tk equals Tk. "Net Sharpe" uses E14's cost rule
(18 bp charged on changes of predicted class); G′ charges 18 bp at every event.
Net Sharpe alone, and the AUC and G rates for B0′ and C (not run in the
permutation stage), are in `results/inbox7/`.

| Rate | A-BTC | A-ETH | A-SOL | B0 | B1 | B2 |
|---|---|---|---|---|---|---|
| AUC > 0.55, not purged | 25.5 | 29.0 | 31.0 | 34 | 57 | 85 |
| AUC > 0.55, purged | 20.5 | 29.5 | 28.0 | 28 | 54 | 82 |
| G (AUC > 0.55 and net Sharpe > 0), not purged | 14.5 | 18.0 | 17.0 | 21 | 44 | 80 |
| G, purged | 12.0 | 20.5 | 17.5 | 18 | 42 | 77 |
| G′ (net Sharpe > 0 with cost at every event), not purged | 20.5 | 31.5 | 36.5 | 25 | 54 | 81 |
| G′, purged | 22.5 | 27.5 | 34.0 | 22 | 51 | 75 |
| T1 and net Sharpe > 0 | 9.0 | 10.0 | 11.0 | 16 | 35 | 71 |
| T1 and G′ | 6.0 | 9.5 | 10.5 | 14 | 31 | 65 |
| T2 and net Sharpe > 0 | 9.0 | 9.0 | 10.0 | 12 | 35 | 72 |
| T2 and G′ | 5.5 | 7.0 | 8.0 | 8 | 29 | 62 |
| T3 and net Sharpe > 0 | 3.5 | 4.0 | 3.5 | 6 | 24 | 47 |
| T3 and G′ | 2.5 | 3.0 | 2.5 | 4 | 19 | 42 |
| T4 and net Sharpe > 0 | 5.5 | 5.0 | 5.0 | 9 | 22 | 50 |
| T4 and G′ | 4.0 | 4.0 | 4.0 | 6 | 19 | 45 |

**What this means for E14.** These set the scope of E14's conclusions; none is
withdrawn.
- **PBO on refitted CPCV splits (S1, S4).** E14 concluded that PBO, computed
  its way, is not a test for an edge. That holds for the construction it names,
  refitted CPCV splits with in-sample measured on the fitted training events,
  and it now holds in two model families. In both, PBO < 0.50 passes fewer
  series as the planted effect grows (CatBoost 48, 17 and 6%; LightGBM 54, 36
  and 25%), and mean PBO rises (0.510 to 0.712; 0.486 to 0.585).
- **PBO under canonical CSCV and the validation slice (S3, S2).** Here PBO moves
  with the planted effect in the three cells with power. Canonical CSCV passes
  13, 26 and 46% at 0.20, and 54, 70 and 84% at 0.50. The validation slice
  passes 47, 64 and 71% at 0.50; at 0.20 it passes 0 to 3% and has no power.
- **PBO thresholds are not test levels.** Without a planted effect, canonical
  CSCV passes 13.0–15.5% of demeaned series at 0.20 (B0, A-BTC, C) and 23% with
  the mean kept (B0′); at 0.50 it passes 54.0–54.5% (B0′ 59%). E14's guidance
  for V4 stands: report PBO with the grid and the in-sample measure, as a
  description of the search, not as a pass-or-fail gate. This entry adds: name
  the construction (refitted CPCV or canonical CSCV).
- **What drives the difference is untested.** S2 and S3 differ from S1 in more
  than the in-sample measure: S2 fits on fewer events, and S3 uses six-fold
  predictions without refitting per split. E14 observed that in-sample selection
  on fitted events picks one of the two depth-6, learning-rate-0.1
  configurations on 94–99% of splits. Whether in-sample selection on
  fitted events causes the opposite direction of S1 and S4 was not tested.
- **Drift.** E14's signal-free PBO < 0.50 rates (31.5–32.5%) were measured on
  series that keep each asset's mean return. On the seeds of the planted arms,
  S1 passes 26% of series with the mean kept (B0′) and 48% with it removed (B0);
  paired by seed, exact McNemar p = 0.002, a comparison outside the
  pre-registration. E14
  compared its planted arms (21% and 8%) with signal-free series on other seeds.
  Their matched control is B0′, at 26%.
- **Permutation test.** E14 named two untested candidates for the excess size.
  - Purging the holdout moves A by at most one point: T1 to T2 goes from 15.0,
    15.0 and 15.5% to 14.0, 14.5 and 15.5% (B0 25 to 19%, B2 74 to 75%).
  - Keeping the dependence between neighbouring labels lowers the rate, to
    6.0–6.5% (T3) and 7.0–8.5% (T4). It also lowers power: B2 passes 75% under
    T2, 48% under T3 and 50% under T4.
  - Removing the mean does not remove the excess: T1 passes 15.0–15.5% of
    demeaned series, against E14's 14.0–17.5% with drift kept.
  - These comparisons are descriptive, as pre-registered, and do not establish
    a cause.
- **V6** stands with E14's qualification. In this pipeline a full shuffle of
  overlapping labels runs above its nominal size, and a dependence-preserving
  permutation runs closer to it at a cost in power.
- **AUC alone.** On demeaned series, AUC > 0.55 flags 25.5, 29.0 and 31.0% (BTC,
  ETH, SOL; holdout not purged), against E13's 27.0, 30.0 and 26.5% with drift
  kept.

**Other observations, outside the decision rules.**
- Arm C keeps serial dependence within blocks, and the base model finds
  structure in it: mean holdout AUC 0.536, AUC > 0.55 on 43.5% of series. C is a
  null for drift, not for predictability. Its PBO rates are close to A's, except
  S4 at 0.50 (63.5% against 49.5%).
- B0 and A-BTC come from the same generator on different seeds. B0 passes more
  often on several statistics, for example T1 at 25% against 15.0% (two-sided
  Fisher p = 0.04, one of many unplanned comparisons). We do not interpret this.
  The decision rules use B0 for PBO size and A for permutation size, as fixed in
  advance.

**Use of the outcomes.** The pre-registration ties the outcomes to the decision
on a further paper: a full measurement paper if both outcomes are informative,
a shorter paper if one is, none if neither is. It does not define
"informative". We take the conservative reading: the PBO outcome is
informative and the permutation outcome ("partial") is not, so the outcomes
support at most a shorter paper on PBO. Whether to write it will be decided
after a search of prior work on the same question. This search was not
pre-registered.

**Limits.**
- One grid per model family (eight configurations), and one planted mechanism,
  which is close to one of the features. β was not recalibrated after removing
  the mean (mean holdout AUC 0.558 and 0.604).
- The power arms are BTC, 100 series each. Every arm is daily. An intraday null
  is outside this study, so V5 remains unresolved.
- No intervention tests which design difference drives the PBO contrast, or
  whether label dependence causes the excess size.
- Under exact exchangeability, the pass rule (real AUC above the interpolated
  95th percentile of 100 permuted AUCs) passes about 5.9% of series, not 5%. The
  size thresholds were fixed with this rule.
- Neither T3 nor T4 is an exact randomization test when labels are serially
  dependent; T3's offsets exclude shifts close to zero.
- Three verdicts are within two series of their thresholds: T3 power is two
  series short of 50%, T4 power is exactly 50%, and S2 at 0.20 would be
  inconclusive rather than no power with two more B2 passes.

**What this study does not change** (pre-registration §5). The checklist
wording, the list of results that still stand in E13, and E14's published
numbers. SSRN v3.3 predates this entry.

**Supersedes.**
- E14's sentence "Canonical CSCV and a validation-slice in-sample measure were
  not run."
- E14's description of the two permutation candidates as untested: both were
  compared descriptively here; neither is established as the cause.
- E14's KNOWN_ISSUES 6 bullet ("neither threshold makes PBO a test for an edge
  in this pipeline") and `KNOWN_ISSUES.md` #6 ("at neither threshold is PBO a
  test for an edge in this pipeline"): both hold for refitted CPCV splits with
  in-sample measured on the fitted training events (S1, S4), not for canonical
  CSCV (S3) or, at 0.50, a validation-slice in-sample measure (S2).
- SSRN v3.3, where it says that canonical CSCV and a validation-slice
  in-sample measure were not run, or that their behaviour is open, or that the
  two permutation candidates, a dependence-preserving permutation and a purged
  holdout, were not tested.

## E16: The 340-variant corpus re-counted; E13's corpus figures restated

**How it surfaced.** On 21 October 2026, during proofreading of text that cites
E13's corpus figures, rows were found in `results/reference/variants_340.csv`
that are not strategy variants at 18 bp. Each kind of row was then traced to the
code that wrote it. E13 says of every result it lists as still standing that
"the definition in the paper, the computation in the code and the stored value
agree". For the corpus count this does not hold. As E13 states, the stored
values reproduce from their inputs, but not every row counted is a variant at
18 bp, and some results are counted more than once.

**What can be cited after E16.**
- A majority of the strategy variants in the archive have a negative net Sharpe
  ratio at 18 bp under every definition below; 165 of 278 under the corrected
  count.
- 15-minute 9 of 9 and 4-hour 4 of 9 under every definition; 1-hour 98% and
  daily 26% under the corrected count.
- The 1h tree models' long share: 79.5–98.4% without class weighting,
  32.3–52.8% with it.
- The order-flow run: net Sharpe ratio 0.135 against a gross 0.750 at 101
  trades a year, from a published row; 82% as a ratio of those two values, not
  as a stored value; its permutation pass with the qualifications below.
- The Deflated Sharpe Ratio: no variant passes, but no stored row could pass
  the gate as computed, so the result carries no evidence about multiple
  testing.

**What the 340 rows contain.** Rows that are not, or not separately, strategy
variants at 18 bp:

| Rows | Count | Negative in `net_sr_18bp` | What they are |
|---|---|---|---|
| Cost sweep at 0, 5, 10, 30 and 50 bp | 15 | 1 | The cost-sensitivity rows of three signals, stored under `net_sr_18bp` although their cost is not 18 bp |
| Synthetic signals at 18 bp | 3 | 0 | `COST_ML_1h_18bp`, `COST_ML_15m_18bp` and `BASE_1h ML balanced`: the random-signal proxies withdrawn in E2 |
| `ABLATION_` | 12 | 9 | Refits of the three `EXT_BTC_1h` models with checklist items switched on or off (see below) |
| Same result, every stored value equal | 28 | 0 | 12 funding-rate rows and 16 gold-threshold rows (see below) |
| Same result under a different label | 4 | 3 | `BB_{BTC,ETH,SOL}_CatBoost` and `COST_benchmark_18bp` (see below) |

- **Funding-rate rows (12).** In the model step, a feature missing on half of
  the event rows or more is dropped without a log line. The added feature of
  four of the five funding-rate variants was dropped this way, so their models
  are the base models. Each of these rows repeats `EXT_BTC_1d_*` for the same
  model. The variant built on daily Binance funding is the one whose feature
  entered the model.
- **Gold-threshold rows (16).** The rule filters named `SHAP_g…` require a
  minimum gold return. On this sample, the thresholds −0.07 and −0.10 select the
  same signal days under every setting of the other two filters, so each of the
  16 pairs holds one result.
- **`BB_{asset}_CatBoost` (3).** A second script re-ran the configuration of
  `EXT_{asset}_1h_CatBoost`. The two runs differ by less than 1e-8 in net Sharpe
  ratio. The stored Sharpe ratio, AUC, long shares and event count are equal;
  the `BB_` rows leave `pbo` blank.
- **`COST_benchmark_18bp` (1).** It repeats `BASE_momentum_benchmark`: same
  signal, same cost.
- **`ABLATION_` (12).**
  - These rows refit the three `EXT_BTC_1h` models with class weighting, CPCV,
    costs and the permutation test switched on or off, and score them one bar
    ahead instead of four.
  - The CPCV and permutation switches change no stored number, except that
    switching CPCV off leaves `pbo` blank. The 12 rows therefore hold six
    distinct results from four fitted models.
  - Three rows (`No_Cost`, `No_Cost_No_Perm` and `AUC_Only`) store a Sharpe
    ratio before costs under `net_sr_18bp`.
  - The BTC 1h CatBoost fit therefore appears in the corpus with two values at
    18 bp: −1.6349 (one bar ahead) and −0.8767 (four bars ahead).

**Definitions.** Each is computed from `variants_340.csv` by
`scripts/corpus_definitions.py`. The Deflated Sharpe Ratio is computed as in
E13, with N equal to the number of rows of the definition. The ladder counts
the rows that `scripts/derive_tf_ladder.py` counts.

| Definition | Rows | Negative at 18 bp | 15m | 1h | 4h | Daily | DSR benchmark / best SR / best DSR / passes |
|---|---|---|---|---|---|---|---|
| As printed in E13 | 340 | 178 (52.4%) | 9/9 | 128/133 | 4/9 | 17/77 | 2.73 / 1.98 / 0.33 / 0 |
| Without the 18 `COST_` rows (the footnote in "Corpus composition") | 322 | 177 (55.0%) | 9/9 | 128/133 | 4/9 | 17/77 | 2.71 / 1.98 / 0.33 / 0 |
| … and rows with the same asset, frequency and model label, equal in every stored result except a blank `pbo`, merged | 288 | 172 (59.7%) | 9/9 | 123/127 | 4/9 | 17/65 | 2.68 / 1.98 / 0.33 / 0 |
| Corrected: 18 bp, no synthetic signal, no `ABLATION_`, each result once | 278 | 165 (59.4%) | 9/9 | 116/118 | 4/9 | 17/65 | 2.67 / 1.98 / 0.33 / 0 |

- **The corrected count is the author's reading.** It reads the paper's wording
  as strategy variants at 18 bp, with each result counted once. It was settled
  after the numbers above were known.
- **Three choices are arguable.**
  - All twelve `ABLATION_` rows are excluded, although four of their six
    distinct results are fits at 18 bp scored one bar ahead. Keeping those four
    gives 282 rows, 168 negative (59.6%).
  - The five baselines are kept. Excluding them gives 273 rows, 165 negative
    (60.4%).
  - The eight `FORT_` rows are kept, although each uses later data (see
    Limits). Excluding them gives 270 rows, 163 negative (60.4%).
- **Two earlier versions.** Both were computed before the corrected count was
  settled. Neither changes a conclusion.
  - 282 rows, 168 negative (59.6%): it kept the four rows that repeat a result
    under another label. These are the same totals as keeping the four
    `ABLATION_` results; the 1h rung differs (119 of 121 against 119 of 122).
  - 279 rows, 165 negative (59.1%): it kept `COST_benchmark_18bp`.
- **Removed rows.** The 62 rows removed from the 340 include 13 negative ones,
  so the share rises.
- **Composition of the corrected 278 rows.**
  - 85 are rule-based: 5 baselines, 48 gold filters on the rule benchmark, 24
    cross-sectional configurations and 8 `FORT_` rows. 21 of them are negative.
    The `FORT_` rows filter the momentum benchmark by a percentile rank of
    realised volatility. They fit no model, although the corpus table of the
    camera-ready paper and SSRN v3.2 calls them "Volatility prediction
    overlays".
  - 193 are model variants, of which 144 (74.6%) are negative.

**What changes in E13's list of results that still stand.**
- **The negative share.** E13 says "178 of the 340 variants (52%) have a
  negative net Sharpe ratio at 18 bp". The corrected count is 165 of 278 (59%).
  - The direction holds under every definition: a majority is negative.
  - The level is a property of this archive. It depends on which experiments
    the corpus contains: 144 of 193 model variants are negative, against 21 of
    85 rule rows.
- **The ladder.**
  - Unchanged: 15-minute 9 of 9 and 4-hour 4 of 9 (44%).
  - 1-hour: 96% (128 of 133) becomes 98% (116 of 118).
  - Daily: 22% (17 of 77) becomes 26% (17 of 65). E13 calls the ladder rows
    "model variants", but the daily rung also counts the eight rule-based
    `FORT_` rows. Without them it is 15 of 57. The README carries 26%, the rung
    under the filter of `derive_tf_ladder.py`, applied to the corrected rows.
  - The ordering is unchanged.
- **Bull bias.** E13 says that without class weighting "tree models predict
  'long' for 90–97% of test observations (BTC 97.2%, ETH 90.5%, SOL 97.0%)" and
  that weighting brings this to 32–45%. Those six values come from the three
  1h CatBoost rows.
  - Across the nine 1h tree rows of the same experiment, the unweighted long
    share is 79.5–98.4%: CatBoost 90.5–97.2%, LightGBM 79.5–88.7% and random
    forest 95.4–98.4%.
  - With weighting it is 32.3–52.8%, with AUC 0.49–0.53.
  - The other tree rows of the corpus use other labels and frequencies. Across
    the 175 tree rows of the 340 that store a long share, the unweighted share
    ranges from 2.6% to 99.9%.
  - The README's "Weighting removes the bias" becomes "weighting brings the long
    share to 32–53%" (32.3–52.8%).
  - **Deep learning.** E13's range of 58–93% is a range of per-architecture
    means over seven architectures, which E13 does not say.
    - Two of the seven are single rows from a separate bull-bias run: LSTM
      57.7% and SimpleRNN 63.3%.
    - The other five are means over six `DL_` rows each, from TCN 75.3% to MLP
      93.2%.
    - Single rows range from 57.7% to 100%.
    - "[F]or the two MLP rows the 'balanced' column repeats the unbalanced
      value" holds for all 12 MLP rows.
- **The Deflated Sharpe Ratio.** No variant passes under any definition or
  choice above (a benchmark of 2.65–2.73 in the units of the stored Sharpe
  ratios, against a best observed value of 1.98). This carries no evidence
  about multiple testing. With T set to the 20% holdout, no row reaches 0.95
  even against a benchmark of zero: the highest value is 0.92, on 131 holdout
  events. A gate that no stored row can pass does not separate these variants.
- **The order-flow 1h run.**
  - E2 cites `results/round_a_orderflow/backtest_with_costs.csv` as the source.
    That file was never in this repository. It is a four-row summary (sha256
    `dea45b24f6334cff9847f0cb93eb50384bcfa3649c340cdefd1ab0009ec39d5c`) written
    by a private script.
  - The model is fitted on hourly order-flow features. The backtest takes one
    position a day and uses daily returns.
  - Gross Sharpe 0.750, net Sharpe 0.135 and 101 trades a year reproduce from
    the file. Its row for the model, which carries no strategy parameter, is
    now published (`results/reference/order_flow_summary.csv`).
  - The row is consistent with a flat-or-long position charged 18 bp per
    change, 36 bp a round trip, as for the rule rows: its exposure equals its
    long share (0.41), and its total cost of 24.66% equals 137 changes at
    0.18%. The net 0.135 and the 82% depend on that convention.
  - "82%" is 1 − 0.135/0.750. It stands as a ratio of the two stored Sharpe
    ratios, computed in the text, not as a stored value or a code output. The
    file's own cost column is a different quantity: total cost as a share of
    gross return, 83.05%.
  - **The permutation test.**
    - The same script and run wrote the permutation result: real AUC 0.5703, a
      95th percentile of 0.5163 over 200 permutations, and none of the 200
      permuted AUCs at or above the real one. The stored p is 0 (0 of 200). E13
      reports it as p < 0.005, the resolution of 200 permutations.
    - The 200 permuted AUCs were not kept, so these values cannot be recomputed.
    - Each permutation shuffles the whole label vector and refits the model.
      This is the full-shuffle design that E14 and E15 found to run above its
      nominal size in the daily pipeline, where the labels overlap; E15's
      comparisons there are descriptive and establish no cause. Whether this
      run's labels overlap was not traced.
    - The features and hyperparameters were chosen before the test. The
      hyperparameter search's stored best AUC equals the reported holdout AUC
      to full precision. The permutations reuse these choices and do not repeat
      the search, so the p-value does not account for it.
    - The backtest model is built by the same code, data and seed as the model
      of the test, but its predictions were not stored.
    - E13's "passes a 200-permutation test" stands as a stored result,
      qualified by the full-shuffle design and by the search not being
      repeated.

**What changes in this repository.**
- **New files.**
  - `scripts/corpus_definitions.py`, which writes
    `results/reference/corpus_definitions.csv` (the definitions above) and
    `results/reference/corpus_rows.csv` (one line per row of `variants_340.csv`,
    with its kind, its group and its membership in each definition);
  - `results/reference/order_flow_summary.csv`: the model's row of the
    order-flow summary named above.
- **Restated with the corrected count.**
  - the README's "What still stands", including its bull-bias, order-flow and
    DSR bullets;
  - the landing page;
  - the canon list of `scripts/gate_landing.py`, which required "340" and "52%";
  - `scripts/README.md`;
  - the docstring of `scripts/derive_tf_ladder.py`.
- **Benchmark parameters.** `scripts/derive_tf_ladder.py` and
  `scripts/derive_corpus_stats.py` no longer state the rule benchmark's
  parameters.
- **The checklist example.** The example run in `valid/checklist.py` and the
  test `test_paper_example` used the paper's figures:
  - PBO 0.000 and the cost-ladder values 0.640, 0.530 and 0.335 are withdrawn
    (E2, E13).
  - Permutation p = 0.000 was replaced by p < 0.005 (E13).
  - Var(SR_IS) = 0.012 stands as a stored value, but its comparison is
    withdrawn (E13).
  - The gross Sharpe ratio 0.84 matches the mean out-of-sample CPCV Sharpe ratio
    of the best configuration of a different 1h run (0.8411, a private output).
    No stored output gives 0.84 as the order-flow run's gross Sharpe ratio, yet
    the example pairs it with that run's net 0.135.

  The example and the test (now `test_illustrative_example`) use illustrative
  values, labelled as such.
- **`KNOWN_ISSUES.md`.**
  - item 8: the silent feature drop;
  - item 9: the `net_sr_18bp` column holding other cost levels and Sharpe ratios
    before costs.
- **Pointers.** The README's banner, file tree and BibTeX note, the landing
  page's links and the note in `CITATION.cff` point to E16.

**Limits.**
- The corpus is a record of earlier experiments, not a designed sample of
  strategies. Rows are not independent trials: for example, the 48 gold filters
  share one rule benchmark.
- The stored Sharpe ratios are annualised with √252 at every bar frequency
  (KNOWN_ISSUES #4). Signs, and so every count here, do not depend on that
  scale. The DSR uses them as E13 did.
- **Cost convention.** The paper calls 18 bp a round-trip cost. The code
  applies it in three ways:
  - Model rows (`EXT_`, `ML_`, `DL_`, `OF_`, `TB_`, `BB_`): positions of +1 or
    −1, and 18 bp per change of predicted class. That is 18 bp per round trip
    of one unit.
  - Rule rows with flat-or-long positions (baselines, the benchmark, `SHAP_`,
    `FORT_`): 18 bp per change, so a round trip costs 36 bp. Each partial step
    of a gradual `FORT_` overlay pays the full 18 bp.
  - Cross-sectional rows (`XS_`): 20 bp per trade, with trades derived from an
    assumed turnover and counted twice, so 40 bp a round trip, rather than
    from the measured changes.

  Comparisons between rule and model rows inherit these differences.
- The `FORT_` rows rank realised volatility over the full series, so each
  day's rank uses later data.
- The traces from rows to code rest on private scripts from March and April
  2026 that were not under version control.

**Supersedes.**
- **In E13's "What still stands".**
  - the negative share, the ladder and the bull-bias ranges, as restated above;
  - the sentence that definition, code and stored value agree, for those items
    and for the order-flow run;
  - the DSR item as evidence about multiple testing: it stands as a
    computation, but no stored row can pass the gate (highest 0.92 at a
    benchmark of 0);
  - the DSR benchmark 2.73 as the single figure;
  - the order-flow bullet, qualified as above.
- **In the "Corpus composition" section.** The 52% and 178/340 kept "as the
  figures for the corpus as constructed".
- **In E2.**
  - the citation of the order-flow file as a source in this repository;
  - 82% as a value stored in that file;
  - the ladder counts n = 133 and n = 77.
- **In this repository.** The README's "What still stands" and the sentence
  that introduces it, the landing page's corpus sentence, and the two
  `scripts/README.md` lines on the corpus share and the ladder ("which stands",
  "The ordering and the share of negative variants stand").
- **In SSRN v3.3.** Where it states 178 of 340, 52%, the 96% and 22% rungs, or
  82% as a stored value; the 32–45% range and "it removes the bias but not the
  weakness"; the deep-learning range 58–93% as a range of rows; the label
  "model variants" for the ladder; the DSR benchmark 2.73 as the single figure
  and the abstract's DSR clause as evidence about multiple testing; and that
  these results passed the re-verification.

## Unchanged

*Superseded by E13 (October 2026): most of the results listed below are withdrawn.*

The literature audit (Table 3), bull bias (Table 4), the ablation and Monte
Carlo false-positive results (Tables 6–7), the multiple-testing analysis and
Bonferroni survivor table (Table 8, nine survivors, t from 4.07 to 21.70, all
PBO = 1.0, zero surviving the Deflated Sharpe Ratio, E[max SR] = 2.93), and the
corpus-level figures of 52% negative net Sharpe (178/340) and 4.4% exceeding the
benchmark (15/340) were each recomputed from their stored artifacts and match
the printed values. Scripts: `scripts/derive_corpus_stats.py`,
`scripts/derive_bonferroni.py`, `scripts/derive_tf_ladder.py`.

---

Entries E1–E11 were identified by applying the VALID audit process to our own
camera-ready before the workshop, and the version presented at KDD-MLF 2026
includes them. This footer previously said that the conclusions were unchanged.
E13 (October 2026) supersedes that statement: it withdraws most of the paper's
empirical results.

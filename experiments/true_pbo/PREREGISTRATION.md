# Pre-registration of the E14 re-measurement (verbatim record)

This is a verbatim copy of the pre-registration document behind
[CHANGELOG E14](../../CHANGELOG_CR.md#e14-pbo-and-the-permutation-test-re-measured-under-a-pre-registered-design).
It was written in the author's private working repository and committed there,
with amendments A1–A4 and the instruction to run it, on 2026-10-19 at
00:06 KST (private commit c2bfe78), before any of the new statistics (true PBO,
the symmetric permutation test, the planted-signal arm) was computed. The AUC
pass rates on the same signal-free series were already known from E13 and are
quoted in the document. The planted-signal
strengths were committed at 02:57 KST (b7d49ef), before the evaluation seeds
were run. The private repository is not public, so these timestamps cannot be
checked from outside; the document is reproduced here unedited so that the
design can be compared with what was run.

Internal labels in the text:
- *INBOX #n*: a numbered task given to the coding agent that ran the work.
- *rule 43*: the author's working rule that pass criteria are fixed before results are seen.
- *Option I-1 B*, *B2*: internal decision labels for "recompute PBO as defined" and "add a power arm".
- *Lesson α*: the finding that the stored `pbo` column was not PBO (E13, item 2).
- *C3*, *C6b*, *δ*: items of an internal code audit, published as E13 items 5 and 6 and KNOWN_ISSUES 3–4.
- `canon/var_sr_null_corrected*`, `src/paper_extras.py`, `round_1h/…`: files in the private repository.

Where the main table and the amendments differ (for example, annualised versus
per-event Sharpe ratios, or stored versus re-run permutation tests), the
amendments govern; they were added before the run.

---

# Pre-registration: true PBO on the Monte Carlo null (option I-1 B). APPROVED by the user 2026-10-19; amendments A1–A4 added before any run. Executed as INBOX #6.

Fixed before any result is seen (rule 43). Any change after a run starts is logged here with its reason, and the original numbers are reported too.

| Element | Fixed choice | Why |
|---|---|---|
| Null series | BTC, ETH, SOL daily bootstrap, seeds 0–199, the exact inputs of `canon/var_sr_null_corrected*` (AUC FPR reproduces 27.0 / 30.0 / 26.5%) | Same data as the printed results; any difference is the statistic, not the data |
| Pipeline | 17 features, CUSUM events, Triple Barrier labels, v0.2 event-aligned forward returns, CPCV N = 6, k = 2, purge as in `mc_expansion.py` | Unchanged |
| Configuration grid | Two grids, both reported. Grid R (for V5): the three configurations behind the real Var(SR_IS) = 0.012, i.e. `round_1h/optuna_{catboost,lightgbm,rf}.json` as loaded in `src/paper_extras.py` L40–45, L110 (M = 3; INBOX #4 T2). Grid P (for PBO): CatBoost 100 iterations × depth {4, 6} × learning rate {0.03, 0.1} × class weights {none, balanced} (M = 8). PBO from grid R is reported but not used for the gate, because at M = 3 a path counts only when the in-sample best is out-of-sample worst | V5 needs the null and the real value over the same grid; PBO needs enough configurations to rank |
| Performance per configuration and path | Annualised Sharpe of positions (+1 / −1) × event-aligned forward returns, in-sample on the training groups, out-of-sample on the two test groups | The manuscript defines PBO on strategy performance |
| PBO | Eq. 1 exactly: relative rank ω̄ = rank / (M + 1) of the in-sample-best configuration out-of-sample, PBO = share of the 15 paths with logit(ω̄) ≤ 0. Cross-check with `valid.metrics.compute_pbo` | Matches the printed definition |
| Var(SR_IS) | Eq. 2, `valid.metrics.var_sr_is`: variance across configurations within a path, averaged over paths | Matches the real 0.012 |
| Gates reported | PBO < 0.20 (manuscript) and PBO < 0.50 (`valid/checklist.py` default; the two thresholds currently disagree); AUC > 0.55; permutation (stored results, size caveat δ); net SR > 0; every conjunction | All reported, none selected afterwards |
| Power arm (B2) | Same pipeline on BTC series with a planted signal at two strengths (holdout AUC about 0.55 and 0.60), 100 series each; report the pass rate of every gate | Lesson α: a gate that rejects everything has 0% FPR. Size without power cannot support "use this gate" |
| Output | Per-series CSV (asset, seed, config results, PBO, Var(SR_IS), gates), summary JSON with command, git hash, data sha256, runtime | Same canon pattern as INBOX #3 |
| Runtime | INBOX #4 T3: 0.040 s per fit, so grid P ≈ 48 min serial (24–48 min at parallel 2). Grid R adds LightGBM and RF fits, so time it on one series first. Power arm ≈ +30% | Fits in one evening, no night run needed |

Reporting commitment: the PBO gate's false-positive rate and power are printed whatever they are, including a result that weakens V4.

## Amendments fixed at approval, before any run (2026-10-19)

| # | Amendment | Why (from INBOX #5) |
|---|---|---|
| A1 | All Sharpe ratios used for ranking, for Var(SR_IS) and for the net-SR gate are **per-event** (mean/sd of position × event-aligned forward return), not annualised. Annualised values may be shown for display only, scaled by √(events per year) | √252 on per-event returns was the cost-ladder error (C3). Var(SR_IS) scales with the square of the factor, so the real and null sides must share one unit |
| A2 | Power arm mechanism: after the i.i.d. bootstrap, a drift is added sequentially: r′(t+1) = r(t+1) + β·σ·z(t). Here σ is the bootstrap return SD, and z(t) is the 5-day return z-score computed from r′ up to t (no look-ahead). β is calibrated on seeds 2000–2019 only, to a mean holdout AUC of about 0.55 and about 0.60, and written down before seeds 1000–1099 are run | A planted signal of known strength, so that each gate's power is measured, not only its size |
| A3 | The permutation test is symmetric: 100 shuffles per series, the same model, parameters and iterations as the real fit, pass if real AUC > 95th percentile of the shuffled AUCs. Applied in the null and the power arms | The stored test used 20 shuffles and weaker permutation models (C6b) |
| A4 | Real-data arms. R1: the real Var(SR_IS) is recomputed with grid R and the A1 definition on the round_1h pipeline. R2: true PBO with grid P on the real 1h pipeline and on BTC daily | A null comparison is only valid if the real value uses the same definition and units |

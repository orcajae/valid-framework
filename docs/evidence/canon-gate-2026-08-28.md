# Canon gate — evidence, 2026-08-28

Branch `feat/canon-gate`, worktree `../canon-gate`. Not pushed.

This file is a log, not an assessment. Commands and their output only.

---

## 1. GATE 1 measurement (before)

Scan: every tracked file, binaries excluded by a NUL/UTF-8 probe.
88 text files before, 90 after (canonical.json and forbidden.json are new).
Scanner: `scan_gate1.py`, kept out of the repo (session scratch).

A first pass used `(?![\d.,])` as the trailing boundary and silently missed
every value followed by a comma -- `"total_variants": 340,` in
results/reference/multiple_testing.json among them. The counts below are from
the corrected pass.

| token | before hits | before files | after hits | after files |
|---|---|---|---|---|
| `340` | 80 | 19 | 95 | 20 |
| `322` | 1 | 1 | 4 | 3 |
| `18` | 105 | 31 | 120 | 33 |
| `80` | 39 | 22 | 60 | 23 |
| `74` | 66 | 21 | 81 | 21 |
| `52%` | 16 | 10 | 16 | 10 |
| `4.4%` | 19 | 10 | 22 | 10 |
| `2.5` | 54 | 16 | 55 | 16 |
| `9 of 12` | 10 | 8 | 13 | 9 |
| `27%` | 9 | 7 | 10 | 8 |
| `0/74` | 4 | 3 | 5 | 4 |
| `26,215` | 0 | 0 | 4 | 2 |
| `6,518` | 0 | 0 | 4 | 2 |
| `1.98` | 17 | 10 | 20 | 11 |
| `2.93` | 4 | 4 | 6 | 5 |
| `6508779` | 18 | 11 | 22 | 13 |
| `f2a901b8` | 0 | 0 | 2 | 1 |

`26,215`, `6,518` and `f2a901b8` had zero hits in the tracked text before this
work. GATE 1.5 then read the six tracked PDFs with `pdftotext -layout`:

```
poster/kddmlf2026_valid_poster_900x1050.pdf        18296 chars   retired tokens: 0
poster/kddmlf2026_valid_poster_900x1050_FINAL.pdf  18296 chars   retired tokens: 0
poster/kddmlf2026_valid_poster_900x1600.pdf        19401 chars   L155: @jwquant
docs/assets/valid_checklist_onepage.pdf             2448 chars   retired tokens: 0
releases/backtesting_checklist.pdf                  2448 chars   retired tokens: 0
docs/paper/Paper18_VALID_camera_ready_corrected.pdf 70107 chars  retired tokens: 0
```

`26,215` and `6,518` are absent from the PDFs as well. `f2a901b8` resolved:

```
$ md5 -q docs/paper/Paper18_VALID_camera_ready_corrected.pdf
f2a901b81576f59dc114e7322734ddcc
```

Two duplicate-asset facts recorded and not acted on:

```
$ md5 -q docs/assets/valid_checklist_onepage.pdf releases/backtesting_checklist.pdf
b6feb490c0d3db85cef65b07e6aa60ae
b6feb490c0d3db85cef65b07e6aa60ae

$ md5 -q poster/kddmlf2026_valid_poster_900x1050.pdf poster/kddmlf2026_valid_poster_900x1050_FINAL.pdf
e024c6a72352d97a928fe378e5e00d84
c79486ed55ef6677c8c65bf9b796e934
```

The two posters extract identical text (18296 chars each) and differ in bytes.
Not investigated; not changed.

The camera-ready carries a denominator set that appears in no tracked text file:

```
$ pdftotext -layout docs/paper/Paper18_VALID_camera_ready_corrected.pdf - | sed -n '189,190p'
them. Excluding
them, 55.0% of the remaining 322 variants are negative net of costs
```

---

## 2. canonical.json

```
$ python3 scripts/derive_canonical.py
wrote canonical.json: 40 derived, 5 manual (2 unresolved), generated_at=2026-08-27T16:28:40Z
```

- `figures`: 40 keys, all derived
- `manual`: 5 keys, none derived -- `paper_number`, `ssrn_id`,
  `self_audit_score`, `n_trades_total`, `n_trades_oos`
- of those, 2 carry `"value": null` -- `n_trades_total` and
  `n_trades_oos`, marked `"scope": "track_a_external"`, excluded from all three
  checks per the 2026-08-28 ruling
- `unavailable_sources`: 4 entries -- sources named in the directive
  that this tree does not contain

Literal audit of the generator (only the manual section may hold a number):

```
$ grep -nE '"value": (18|26215|6518)' scripts/derive_canonical.py
372:            "value": 18,
exit=0
```

---

## 3. Deliberate failures

Each check was broken on purpose and restored with `git checkout --`.
Exit codes are distinct by design: A=2, B=3, C=4.

```
### FAIL 재현 A-1 — 원본(variants_340.csv) 1행 추가
$ ./scripts/gate_canon.sh a
------------------------------------------------------------
CHECK A — canonical figures match a fresh derivation
------------------------------------------------------------
FAIL figures: 5 key(s) drifted
     - n_variants
         committed : {"value": 340, "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "340"}
         derived   : {"value": 341, "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "341"}
     - n_variants_ex_cost
         committed : {"value": 322, "derived_from": ["n_variants", "n_cost_entries"], "op": "subtract", "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "322"}
         derived   : {"value": 323, "derived_from": ["n_variants", "n_cost_entries"], "op": "subtract", "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "323"}
     - pct_negative_ex_cost
         committed : {"value": 55.0, "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "55.0%"}
         derived   : {"value": 54.8, "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "54.8%"}
     - pct_negative_full
         committed : {"value": 52.4, "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "52%"}
         derived   : {"value": 52.2, "source": "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "52%"}
     - tf_ladder
         committed : {"value": {"15m": {"n": 9, "median_net_sr": -1.8, "best_net_sr": -1.33, "pct_negative": 100}, "1h": {"n": 133, "median_net_sr": -1.32, "best_net_sr": 0.32, "pct_negative": 96}, "4h": {"n": 9, "median_net_sr": 0.28, "best_net_sr": 0.57, "pct_negative": 44}, "1d": {"n": 77, "median_net_sr": 0.54, "best_net_sr": 1.98, "pct_negative": 22}}, "source": "scripts/derive_tf_ladder.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "15m n=9 median=-1.80; 1h n=133 median=-1.32; 4h n=9 median=+0.28; 1d n=77 median=+0.54"}
         derived   : {"value": {"15m": {"n": 9, "median_net_sr": -1.8, "best_net_sr": -1.33, "pct_negative": 100}, "1h": {"n": 133, "median_net_sr": -1.32, "best_net_sr": 0.32, "pct_negative": 96}, "4h": {"n": 9, "median_net_sr": 0.28, "best_net_sr": 0.57, "pct_negative": 44}, "1d": {"n": 78, "median_net_sr": 0.54, "best_net_sr": 1.98, "pct_negative": 22}}, "source": "scripts/derive_tf_ladder.py:main (results/reference/variants_340.csv)", "derived": true, "print_precision": "15m n=9 median=-1.80; 1h n=133 median=-1.32; 4h n=9 median=+0.28; 1d n=78 median=+0.54"}
PASS manual: 5 key(s) identical
PASS unavailable_sources: 4 key(s) identical
PASS audit coding sheet == mirror  (f30d28f3a0e6d36b31b88feb47c4ea09)
exit=2

$ git checkout -- results/reference/variants_340.csv   # 원상복구

### FAIL 재현 A-2 — audit 코딩시트와 mirror 불일치 (판정 1)
$ ./scripts/gate_canon.sh a
------------------------------------------------------------
CHECK A — canonical figures match a fresh derivation
------------------------------------------------------------
FAIL figures: 1 key(s) drifted
     - audit_csv_mirror_md5
         committed : {"value": "f30d28f3a0e6d36b31b88feb47c4ea09", "source": "md5(results/reference/literature_audit_80.csv)", "derived": true, "print_precision": "f30d28f3a0e6d36b31b88feb47c4ea09"}
         derived   : {"value": "863614cd18f6fa2da23c8cd720e15b7a", "source": "md5(results/reference/literature_audit_80.csv)", "derived": true, "print_precision": "863614cd18f6fa2da23c8cd720e15b7a"}
PASS manual: 5 key(s) identical
PASS unavailable_sources: 4 key(s) identical
PASS audit coding sheet == mirror  (f30d28f3a0e6d36b31b88feb47c4ea09)
exit=2

$ git checkout -- results/reference/literature_audit_80.csv   # 원상복구

### FAIL 재현 B-1 — 폐기 토큰 재유입
$ ./scripts/gate_canon.sh b
------------------------------------------------------------
CHECK B — retired tokens in tracked content
------------------------------------------------------------
whitelist (scanned files excluded, printed rather than applied quietly):
  - forbidden.json
file groups (scope of the landing-only rules):
  landing_raw: docs/index.html, docs/scorer.js, README.md
  landing_visible: docs/index.html, docs/scorer.js
exemptions applied (printed, never silent):
  pbo_zero not enforced in CHANGELOG_CR.md — CHANGELOG_CR.md records the retraction of this very claim and quotes it to do so
  first_empirical_confirmation not enforced in CHANGELOG_CR.md — CHANGELOG_CR.md records the retraction of this very claim and quotes it to do so

FAIL B-1 hard tokens: 2 hit(s)
  [trades_16352] README.md:182: The backtest covered 16,352 trades. Follow @jwquant for updates.
  [handle_jwquant] README.md:182: The backtest covered 16,352 trades. Follow @jwquant for updates.
WARN B-2 context tokens: 123 hit(s) across 16 rule(s) -> gate_warn.log (does not fail the gate)
  n75_25: 3
  n75_64: 1
  n75_of: 1
  n75_zero: 2
  n_107: 5  [known FP]
  pct_33: 6
  pct_53: 6
  pct_72: 3
  pct_band_55_91: 72  [known FP]
  score_10_12: 2
  sr_0186: 2
  sr_0416: 1
  sr_0530: 4  [known FP]
  sr_0640: 3  [known FP]
  sr_095: 9
  sr_neg0222: 3
exit=3

$ git checkout -- README.md   # 원상복구

### FAIL 재현 C — attestation 만료
$ ./scripts/gate_canon.sh c
------------------------------------------------------------
CHECK C — cascade coverage
------------------------------------------------------------
canonical.json generated_at = 2026-08-27T16:28:40Z

PASS README.md                          8 figure(s) current
PASS landing page (docs/index.html)     8 figure(s) current
PASS landing scorer (docs/scorer.js)    4 figure(s) current
PASS CHANGELOG_CR.md                    10 figure(s) current
PASS coding sheet guide (audit/coding_guide.md) 2 figure(s) current
PASS scripts/README.md                  4 figure(s) current
PASS REPRODUCE.md                       4 figure(s) current
NOTE ERRATUM                            no ERRATUM document in this repository; CHANGELOG_CR.md carries the correction history
FAIL SSRN preprint 6508779              never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'SSRN preprint 6508779' "<your name>"
FAIL printed poster 900x1050 v5         attestation 2026-08-01T00:00:00Z predates 2026-08-27T16:28:40Z
FAIL LinkedIn posts                     never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'LinkedIn posts' "<your name>"
FAIL talk script and deck               never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'talk script and deck' "<your name>"
FAIL Claude Code project memory         never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'Claude Code project memory' "<your name>"

notices (recorded, not passed silently):
  - ERRATUM: no ERRATUM document in this repository; CHANGELOG_CR.md carries the correction history
exit=4

$ ./scripts/gate_canon.sh attest 'printed poster 900x1050 v5' '...'   # 원상복구


### FAIL 재현 A-2 (재실행: fresh 값 대조로 수정 후)
$ ./scripts/gate_canon.sh a
------------------------------------------------------------
CHECK A — canonical figures match a fresh derivation
------------------------------------------------------------
FAIL figures: 1 key(s) drifted
     - audit_csv_mirror_md5
         committed : {"value": "f30d28f3a0e6d36b31b88feb47c4ea09", "source": "md5(results/reference/literature_audit_80.csv)", "derived": true, "print_precision": "f30d28f3a0e6d36b31b88feb47c4ea09"}
         derived   : {"value": "863614cd18f6fa2da23c8cd720e15b7a", "source": "md5(results/reference/literature_audit_80.csv)", "derived": true, "print_precision": "863614cd18f6fa2da23c8cd720e15b7a"}
PASS manual: 5 key(s) identical
PASS unavailable_sources: 4 key(s) identical
FAIL audit coding sheet and its mirror diverged
     audit/literature_audit_80.csv             f30d28f3a0e6d36b31b88feb47c4ea09
     results/reference/literature_audit_80.csv 863614cd18f6fa2da23c8cd720e15b7a
exit=2
```

---

## 4. Passing state

```
$ ./scripts/gate_canon.sh all
------------------------------------------------------------
CHECK A — canonical figures match a fresh derivation
------------------------------------------------------------
PASS figures: 40 key(s) identical
PASS manual: 5 key(s) identical
PASS unavailable_sources: 4 key(s) identical
PASS audit coding sheet == mirror  (f30d28f3a0e6d36b31b88feb47c4ea09)
------------------------------------------------------------
CHECK B — retired tokens in tracked content
------------------------------------------------------------
whitelist (scanned files excluded, printed rather than applied quietly):
  - forbidden.json
file groups (scope of the landing-only rules):
  landing_raw: docs/index.html, docs/scorer.js, README.md
  landing_visible: docs/index.html, docs/scorer.js
exemptions applied (printed, never silent):
  pbo_zero not enforced in CHANGELOG_CR.md — CHANGELOG_CR.md records the retraction of this very claim and quotes it to do so
  first_empirical_confirmation not enforced in CHANGELOG_CR.md — CHANGELOG_CR.md records the retraction of this very claim and quotes it to do so
  hard_fail not enforced under docs/evidence/ — evidence logs exist to quote what a check found; a gate that cannot be reported on without failing produces no evidence. This is the one directory-wide hole in B-1 and CHECK B prints it on every run.

PASS B-1 hard tokens: 0 hits across 90 tracked text files
WARN B-2 context tokens: 128 hit(s) across 16 rule(s) -> gate_warn.log (does not fail the gate)
  n75_25: 3
  n75_64: 1
  n75_of: 1
  n75_zero: 2
  n_107: 8  [known FP]
  pct_33: 6
  pct_53: 6
  pct_72: 3
  pct_band_55_91: 74  [known FP]
  score_10_12: 2
  sr_0186: 2
  sr_0416: 1
  sr_0530: 4  [known FP]
  sr_0640: 3  [known FP]
  sr_095: 9
  sr_neg0222: 3
------------------------------------------------------------
CHECK C — cascade coverage
------------------------------------------------------------
canonical.json generated_at = 2026-08-27T16:28:40Z

PASS README.md                          8 figure(s) current
PASS landing page (docs/index.html)     8 figure(s) current
PASS landing scorer (docs/scorer.js)    4 figure(s) current
PASS CHANGELOG_CR.md                    10 figure(s) current
PASS coding sheet guide (audit/coding_guide.md) 2 figure(s) current
PASS scripts/README.md                  4 figure(s) current
PASS REPRODUCE.md                       4 figure(s) current
NOTE ERRATUM                            no ERRATUM document in this repository; CHANGELOG_CR.md carries the correction history
FAIL SSRN preprint 6508779              never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'SSRN preprint 6508779' "<your name>"
PASS printed poster 900x1050 v5         attested 2026-08-27T16:35:52Z by GATE 1.5 pdftotext sweep (Claude, executor)
FAIL LinkedIn posts                     never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'LinkedIn posts' "<your name>"
FAIL talk script and deck               never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'talk script and deck' "<your name>"
FAIL Claude Code project memory         never attested against this canonical.json
       run: scripts/gate_canon.sh attest 'Claude Code project memory' "<your name>"

notices (recorded, not passed silently):
  - ERRATUM: no ERRATUM document in this repository; CHANGELOG_CR.md carries the correction history
------------------------------------------------------------
CHECK A PASS (exit 0)
CHECK B PASS (exit 0)
CHECK C FAIL (exit 4)
exit=4
```

```
$ python3 scripts/gate_landing.py
------------------------------------------------------------
CHECK B — retired tokens in tracked content
------------------------------------------------------------
whitelist (scanned files excluded, printed rather than applied quietly):
  - forbidden.json
file groups (scope of the landing-only rules):
  landing_raw: docs/index.html, docs/scorer.js, README.md
  landing_visible: docs/index.html, docs/scorer.js
exemptions applied (printed, never silent):
  pbo_zero not enforced in CHANGELOG_CR.md — CHANGELOG_CR.md records the retraction of this very claim and quotes it to do so
  first_empirical_confirmation not enforced in CHANGELOG_CR.md — CHANGELOG_CR.md records the retraction of this very claim and quotes it to do so
  hard_fail not enforced under docs/evidence/ — evidence logs exist to quote what a check found; a gate that cannot be reported on without failing produces no evidence. This is the one directory-wide hole in B-1 and CHECK B prints it on every run.

PASS B-1 hard tokens: 0 hits across 90 tracked text files
WARN B-2 context tokens: 128 hit(s) across 16 rule(s) -> gate_warn.log (does not fail the gate)
  n75_25: 3
  n75_64: 1
  n75_of: 1
  n75_zero: 2
  n_107: 8  [known FP]
  pct_33: 6
  pct_53: 6
  pct_72: 3
  pct_band_55_91: 74  [known FP]
  score_10_12: 2
  sr_0186: 2
  sr_0416: 1
  sr_0530: 4  [known FP]
  sr_0640: 3  [known FP]
  sr_095: 9
  sr_neg0222: 3

[gate 3] canon presence (forbidden.json:canon_required)
  PASS 74 empirical papers              '74'
  PASS 80 surveyed                      '80'
  PASS median 2.5 / 12                  '2.5 / 12'
  PASS 340 variants                     '340'
  PASS 52% net-negative                 '52%'
  PASS 4.4% exceed the benchmark        '4.4%'
  PASS benchmark 0.917                  '0.917'
  PASS self-audit 9 / 12                '9 / 12'

[gate 1+2] banned and retired tokens -> scripts/gate_canon.sh b
  PASS gate_canon.sh CHECK B exit 0

[gate 5] local links resolve
  PASS docs/index.html assets/fonts/inter-regular.woff2
  PASS docs/index.html assets/fonts/inter-semibold.woff2
  PASS docs/index.html assets/fonts/inter-black.woff2
  PASS docs/index.html style.css
  PASS docs/index.html assets/valid_checklist_onepage.pdf
  PASS docs/index.html paper/Paper18_VALID_camera_ready_corrected.pdf
  PASS docs/index.html scorer.js
  PASS README.md      LICENSE
  PASS README.md      REPRODUCE.md
  PASS README.md      REPRODUCE.md
  PASS README.md      docs/assets/valid_checklist_onepage.pdf
  PASS README.md      notebooks/crypto_backtesting_starter_kit.ipynb
  PASS README.md      notebooks/worked_example.ipynb
  PASS README.md      LICENSE

[gate 5b] in-page anchors
  PASS #scorer
  PASS #scorer

ALL PASS  (4/4 gates)
exit=0
```

---

## 5. CI workflow

`act` is not installed on this machine. The workflow steps were run locally in
the same order, with the same continue-on-error semantics and the same verdict
rule, by `run_workflow.sh` (session scratch, not committed):

```
$ bash run_workflow.sh
== step: Re-derive canonical.json ==
  exit 0
== step: Diff against the committed file ==
  exit 0
== step: CHECK A ==
  exit 0
== step: CHECK B ==
  exit 0
== step: CHECK C ==
  exit 4
== step: Upload B-2 warn log ==
  gate_warn.log 16957 bytes
== step: Verdict ==
  CHECK A success
  CHECK B success
  CHECK C failure
  ::error::CHECK C failed — a medium is stale or unattested
  elapsed (gate steps only, no pip): 1.6s
  job exit 1
exit=1
```

Per-check timing, measured with `/usr/bin/time -p`:


---

## 6. Hardcoding that remains

This work added a place to check the figures against. It did not remove a single
copy of them. The per-file delta:

| file | before | after | delta |
|---|---|---|---|
| `canonical.json` | 0 | 82 | +82 |
| `scripts/derive_canonical.py` | 0 | 24 | +24 |
| `scripts/gate_landing.py` | 9 | 0 | -9 |

Reading the delta:

- `canonical.json` +82 -- the single home. One authoritative copy, by design.
- `scripts/derive_canonical.py` +24 -- three numeric literals (all in the
  `manual` section) plus source-path strings the scanner counts because it does
  not strip filenames. CHECK B does strip them; this scanner is the GATE 1 tool
  and was left as measured.
- `scripts/gate_landing.py` -9 -- the only duplication actually removed. Its
  BANNED/RETIRED/CANON regexes now live in forbidden.json.

Every other file is unchanged. Specifically still hardcoded and unchanged:

| fact | asserting files |
|---|---|
| 340 variants | CHANGELOG_CR.md, README.md, REPRODUCE.md, docs/index.html, experiments/run_variant_grid.py, notebooks/crypto_backtesting_starter_kit.ipynb, poster/gen_poster.py, poster/gen_poster_900x1050.py, scripts/README.md, scripts/derive_bonferroni.py, scripts/derive_corpus_stats.py, scripts/derive_tf_ladder.py, scripts/gen_fig1_bullbias_340.py, scripts/gen_fig2_340.py, scripts/gen_fig6_340.py |
| 18 bp | CHANGELOG_CR.md, README.md, REPRODUCE.md, notebooks/, poster x2, scripts/README.md, scripts/derive_tf_ladder.py, scripts/derive_corpus_stats.py, scripts/gen_fig2_340.py, scripts/gen_fig6_340.py, examples/worked_example.py, experiments/run_variant_grid.py, experiments/run_multiple_testing.py, valid/costs.py, valid/checklist.py, tests/test_checklist.py |
| 74 empirical | CHANGELOG_CR.md, README.md, audit/coding_guide.md, audit/audit_analysis.py, docs/index.html, docs/scorer.js, poster x2, poster/assets/figure_data.json, scripts/gen_checklist_onepage.py, scripts/gen_fig7_valid_heatmap.py |
| 80 surveyed | CHANGELOG_CR.md, README.md, audit/coding_guide.md, docs/index.html, poster x2, scripts/gen_fig7_valid_heatmap.py |
| 2.5 / 12 | CHANGELOG_CR.md, README.md, docs/index.html, docs/scorer.js (`var MEDIAN = 2.5`), poster/assets/figure_data.json, poster x2, scripts/gen_checklist_onepage.py |
| 52% / 4.4% | CHANGELOG_CR.md, README.md, REPRODUCE.md, docs/index.html, poster x2, scripts/README.md |
| 27% [21, 34] | README.md, REPRODUCE.md, docs/index.html, docs/scorer.js, poster x2, results/reference/REPRODUCE_SUMMARY.md |
| 9 / 12 | CHANGELOG_CR.md, README.md, docs/index.html, docs/scorer.js, poster x2, scripts/gen_checklist_onepage.py |
| SSRN 6508779 | CITATION.cff, README.md, docs/index.html, docs/scorer.js, notebooks x2, poster x2, pyproject.toml, scripts/gen_checklist_onepage.py, valid/multiple_testing.py |

Of these, the media listed in cascade_manifest.json are now checked against
canonical.json by CHECK C. The generators (`gen_fig*.py`, `gen_poster*.py`,
`gen_checklist_onepage.py`), the notebooks, the library defaults
(`valid/costs.py`, `valid/checklist.py`) and the tests are **not** on the
manifest and are not checked by anything.

---

## 7. What this did not solve

1. **No copy was removed.** Sixteen facts still live in ~25 files. The gate
   detects divergence; it does not eliminate the duplication.
2. **The two poster generators remain duplicates.** `poster/gen_poster.py` and
   `poster/gen_poster_900x1050.py` each hardcode every canonical figure
   independently. Out of scope by ruling; a change to one still cannot be
   detected as missing from the other.
3. **Neither poster generator is on the cascade manifest.** They print canonical
   figures and no check reads them.
4. **CHECK C fails today.** Four external media -- SSRN, LinkedIn, the talk deck,
   the assistant memory -- have never been attested against this canonical.json.
   The manifest was seeded with `null` rather than a fabricated attestation, so
   CI is red until a human compares each medium and runs
   `scripts/gate_canon.sh attest`.
5. **`26,215` / `6,518` remain unresolved** in this repo. Recorded with
   `"value": null` and `"scope": "track_a_external"`.
6. **B-2 is advisory only.** 129 warn hits across 16 rules are written to
   `gate_warn.log` and read by nobody automatically. Nothing enforces that a
   human looks.
7. **Two loud exemptions exist.** `pbo_zero` and `first_empirical_confirmation`
   are not enforced in CHANGELOG_CR.md, which quotes both claims to retract
   them. A genuine reintroduction inside that file would pass.
8. **`gate_landing.py` was not reduced to a pure shim.** Its `links`, `anchors`
   and `--live` byte-comparison gates have no equivalent in gate_canon.sh, so
   reducing the file further would have deleted working deployment checks. The
   pattern lists were externalised as ruled; the structural gates stayed.
9. **`regime_conditional.json` is out of reach.** It lives at
   `~/jwquant/paper/kdd-mlf/results/`, outside the repository, so no figure
   derived from it can be gated here.
10. **The `results/reference/literature_audit_80.csv` mirror still exists.**
    CHECK A asserts it equals `audit/literature_audit_80.csv`; it does not
    remove the second copy.
11. **CHECK C's repo-medium check is presence-only.** It asserts the current
    printed value appears in the file. It cannot tell that a *stale* value was
    left beside the current one.
12. **The `900x1600` poster was regenerated, not audited.** `@jwquant` is gone
    because `poster/gen_poster.py` had already been cleaned and the committed
    PDF was a stale build. Nothing else about that PDF was verified.

# Tier 2 sample review (fictional client "Sample Fund A", synthetic data)

SAMPLE: fictional fund, synthetic data, illustrative only. Findings, no assurance.

A demonstration of the deliverable of a Tier 2 review: the false-pass rate (size) and the power of a client's pass-or-fail checks. It is not a study. Its numbers are not research findings and are not to be quoted as such. The report built from these files is [docs/sample-report.html](https://orcajae.github.io/valid-framework/sample-report.html).

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | Design and pass rules, fixed before the run |
| `run.py` | Generator, pipeline, checks and decision rules (numpy, pandas, scipy; no data is read) |
| `results/rows.csv` | One row per series (4 groups × 200) |
| `results/decision.json` | Counts, Wilson intervals, verdicts and descriptive outputs, computed from `rows.csv` |
| `results/lock.json`, `results/run_log.json` | Start record and run log, with the file hashes |

## How it was registered

`PREREGISTRATION.md` and `run.py` were committed to the reviewer's private repository (commit `5d7e0b11bdd729f428a0e662d67f112286d22580`, 2026-10-24 21:04:49 KST) before any registered series was run; the run followed at 21:04:56 KST. That commit cannot be checked from outside the private repository. Both files are published here byte for byte, so their sha256 can be matched to the values recorded in `results/`:

| File | sha256 |
|---|---|
| `PREREGISTRATION.md` | `780eb2cd41e16da2145a771930efda32e849610692ed076aa95fa3b014c42d0c` |
| `run.py` | `d833e3b95613bbbe34e3e2787b2b3ade8be3696c889f12ee07d28a147422486c` |
| `results/rows.csv` | `34d4073c35c9ed31440d1aac97c8a7e033fd959f5c73627bfe33b3144f035568` |

In a real review the client receives the commit hash and both sha256 values before the run.

## Reproduce

```
cd examples/t2_sample
python3 -I -B run.py selftest
python3 -I -B run.py run --out ~/t2_sample_out --workers 2
shasum -a 256 ~/t2_sample_out/rows.csv
```

`run` refuses to start unless both files are committed and unchanged in the clone, and refuses an output folder that already holds a `lock.json`. The new `rows.csv` should match the sha256 above, and the new `decision.json` should equal the published one. Reproduced byte for byte in a fresh clone layout with Python 3.13, numpy 2.5.3, pandas 3.0.5 and scipy 1.18.1. The published run did not record its library versions. Other versions may change rows.csv and, rarely, a selection or a pass (numpy does not promise identical random streams across versions), so compare the counts in `decision.json` as well as the hash.

`PREREGISTRATION.md` is a technical record kept byte for byte, so the wording rules of the public pages do not apply to it.

`scripts/gate_sample_report.py` checks the report page against these files: every number shown is read from `results/`, every result is shown, and `decision.json` is recomputed from `rows.csv`.

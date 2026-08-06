"""Compute literature audit statistics from coding CSV."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from valid.metrics import wilson_ci

CSV = Path(__file__).parent / "literature_audit_80.csv"
df = pd.read_csv(CSV)

# Superseded 2026-08-03. This matched only the `method` column, so id 9
# (Arian, Mobarekeh, Seco 2024) stayed in the empirical set even though its
# `assets` reads "Synthetic" and its `D1_cost` reads "N/A synthetic":
#   empirical = df[~df["method"].str.contains("Survey|Synthetic|PBO|Anomaly|microstructure|DL survey", case=False, na=False)]
NON_EMPIRICAL_IDS = [1, 9, 11, 12, 71, 79]  # see coding_guide.md
empirical = df[~df["id"].isin(NON_EMPIRICAL_IDS)]
N = len(empirical)
print(f"Empirical papers: {N}")

checks = {
    "D1: Cost omitted": empirical["D1_cost"].str.contains("No", case=False, na=False) | (empirical["D1_cost_bp"].astype(str) == "0"),
    "D2: No class balance": empirical["D2_class_balance"].str.contains("Not addressed", case=False, na=False),
    "D3: Random split": empirical["D3_temporal_split"].str.contains("No random", case=False, na=False),
    "D4: Weak validation": empirical["D4_validation"].str.contains("Simple holdout|Random|5-fold", case=False, na=False),
    "D5: BnH only": empirical["D5_baselines"].str.contains("BnH", case=False, na=False) & ~empirical["D5_baselines"].str.contains("MACD|RSI|SMA|momentum|rules|factors|random", case=False, na=False),
    "D6: No net perf": empirical["D6_net_perf"].str.contains("No", case=False, na=False),
    "D7: No code": empirical["D7_code"].str.contains("^No$", case=False, na=False),
    # Reported as a count, not a failure: the row the paper prints as
    # "D4: CPCV used". Emitted here so the printed value has a script behind it.
    "D4: CPCV used": empirical["D4_validation"].str.contains("CPCV", case=False, na=False),
}

rows = []
for name, mask in checks.items():
    r = mask.mean()
    lo, hi = wilson_ci(r, N)
    print(f"  {name}: {mask.sum()}/{N} ({r:.0%}) [{lo:.0%}, {hi:.0%}]")
    rows.append({"dimension": name, "fail_count": int(mask.sum()), "n": N,
                 "rate": r, "ci_lo": lo, "ci_hi": hi})

# The summary artifact the paper's audit table is checked against. It had no
# generator before 2026-08-03 and was carried as a stored file.
SUMMARY = Path(__file__).parent.parent / "results/reference/audit_summary_80.csv"
pd.DataFrame(rows).to_csv(SUMMARY, index=False)
print(f"\nWrote {SUMMARY}")

print(
    "\nNote: live recomputation from the coding sheet under the inclusion "
    "criterion of coding_guide.md (n=74). Every rate above matches the "
    "camera-ready as corrected under E11: D5 is 26/74 (35%) and D7 is 63/74 "
    "(85%). The earlier deviation against a printed 25/75 (33%) and a printed "
    "count of 64 is resolved -- those were the pre-E11 figures."
)

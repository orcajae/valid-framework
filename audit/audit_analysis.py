"""Compute literature audit statistics from coding CSV."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from valid.metrics import wilson_ci

CSV = Path(__file__).parent / "literature_audit_80.csv"
df = pd.read_csv(CSV)

empirical = df[~df["method"].str.contains("Survey|Synthetic|PBO|Anomaly|microstructure|DL survey", case=False, na=False)]
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
}

for name, mask in checks.items():
    r = mask.mean()
    lo, hi = wilson_ci(r, N)
    print(f"  {name}: {mask.sum()}/{N} ({r:.0%}) [{lo:.0%}, {hi:.0%}]")

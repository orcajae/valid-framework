"""Reproduce all results from the VALID paper.
Usage: python experiments/reproduce_all.py
Expected runtime: ~2-4 hours on M4 MacBook Air.
"""
import subprocess, sys, time
from pathlib import Path

SCRIPTS = [
    "01_data_download.py",
    "02_bull_bias.py",
    "03_bull_bias_extended.py",
    "04_lstm_rnn_bias.py",
    "05_stat_econ_disconnect.py",
    "06_witzany_validation.py",
    "07_monte_carlo.py",
    "08_cost_sensitivity.py",
    "09_baselines.py",
    "10_ablation.py",
]

exp_dir = Path(__file__).parent

for script in SCRIPTS:
    path = exp_dir / script
    if not path.exists():
        print(f"SKIP (not found): {script}")
        continue
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED ({elapsed:.0f}s)")
        print(result.stderr[:500])
    else:
        print(f"OK ({elapsed:.0f}s)")
        last_lines = result.stdout.strip().split("\n")[-3:]
        for l in last_lines:
            print(f"  {l}")

print("\nAll experiments complete. Check results/ directory.")

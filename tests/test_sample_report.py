"""The Tier 2 sample report page agrees with its published files.

Runs scripts/gate_sample_report.py: every number on docs/sample-report.html is
read from examples/t2_sample/results, every result is shown, and decision.json
is recomputed from rows.csv. Standard library only; no run is repeated here.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_sample_report_gate():
    r = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts" / "gate_sample_report.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "ALL PASS" in r.stdout

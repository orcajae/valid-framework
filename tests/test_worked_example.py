"""End-to-end regression test for the worked example.

Locks in the qualitative claim the README makes: the mined strategy scores
strictly worse on the VALID checklist than the honest one, and its selected
in-sample Sharpe does not survive out-of-sample.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.worked_example import main


def test_worked_example_smoke(tmp_path):
    summary = main(data="synthetic", smoke=True, out_dir=tmp_path)
    row = {r["strategy"]: r for _, r in summary.iterrows()}
    mined, honest = row["MinedRSI"], row["HonestSMA"]

    # the checklist separates them
    assert mined["score"] < honest["score"]
    # the mined config's holdout net SR collapses below the honest one's
    assert mined["net_sr"] < honest["net_sr"]
    # nothing from the mined grid survives DSR at the 0.95 bar
    assert mined["dsr_best_mined"] < 0.95

    for f in ["report_overfit.md", "report_baseline.md", "summary.csv",
              "fig_is_oos_scatter.png", "fig_equity_curves.png",
              "fig_survivors.png"]:
        assert (tmp_path / f).exists(), f

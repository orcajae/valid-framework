"""VALID: Validation Architecture for Learning-based Investment Decisions."""
__version__ = "0.2.0"

from valid.checklist import VALIDChecker, VALIDReport, VALIDResult
from valid.cpcv import make_groups, cpcv_split, cpcv_paths, run_cpcv
from valid.metrics import (
    annualized_sharpe,
    var_sr_is,
    compute_pbo,
    deflated_sharpe_ratio,
    wilson_ci,
    bootstrap_ci,
)
from valid.multiple_testing import (
    sharpe_pvalues,
    bonferroni,
    holm,
    benjamini_hochberg,
    romano_wolf,
)

__all__ = [
    "__version__",
    "VALIDChecker", "VALIDReport", "VALIDResult",
    "make_groups", "cpcv_split", "cpcv_paths", "run_cpcv",
    "annualized_sharpe", "var_sr_is", "compute_pbo",
    "deflated_sharpe_ratio", "wilson_ci", "bootstrap_ci",
    "sharpe_pvalues", "bonferroni", "holm", "benjamini_hochberg", "romano_wolf",
]

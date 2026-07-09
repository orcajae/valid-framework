"""Shared configuration for all experiments."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "paper" / "figures"

for d in [DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
COST_RETAIL_BP = 18
COST_LEVELS = [0, 5, 10, 18, 30, 50]
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
CPCV_N = 6
CPCV_K = 2
CPCV_PURGE = 20
MC_ITERATIONS = 200
TREE_MODELS = ["catboost", "lightgbm", "random_forest"]
DL_MODELS = ["lstm", "simple_rnn"]


@dataclass
class RunConfig:
    """Pipeline scale knobs. `full` is the documented default tier;
    `smoke` is the CI tier (minutes, pipeline-correctness only)."""
    n_mc: int = 200        # Monte Carlo iterations per setting (seeds 0..n_mc-1)
    n_perm: int = 20       # permutation shuffles per MC iteration
    n_boot: int = 1000     # Romano-Wolf bootstrap replications
    n_bars: int = 2200     # synthetic OHLCV length
    grid: str = "full"     # variant-grid density: "full" | "small"


FULL = RunConfig()
SMOKE = RunConfig(n_mc=10, n_perm=5, n_boot=200, n_bars=1500, grid="small")


def get_run_config(smoke=False):
    return SMOKE if smoke else FULL

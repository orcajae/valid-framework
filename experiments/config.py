"""Shared configuration for all experiments."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "paper" / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
COST_RETAIL_BP = 18
COST_LEVELS = [0, 5, 10, 18, 30, 50]
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
CPCV_N = 6
CPCV_K = 2
CPCV_PURGE = 20
MC_ITERATIONS = 100
TREE_MODELS = ["catboost", "lightgbm", "random_forest"]
DL_MODELS = ["lstm", "simple_rnn"]

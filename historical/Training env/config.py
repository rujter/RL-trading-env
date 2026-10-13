START_SEED = 36
NUM_SEEDS = 1
SEEDS = list(range(START_SEED, START_SEED + NUM_SEEDS))
import pandas as pd

TARGET_CONFIGS = [
    (0.75, 1.30),
    (0.75, 1.30),
    (0.50, 2.20),
    (1.25, 1.90),
]
LOOKBACK = 100
ARCHITECTURE = [
    "def", "2dcnn", "gru"
]
FIXED_LAMBDA = 0.0
df1 = pd.read_csv("cache.txt", header=[0,1], skiprows=[2], index_col=0)

DATAFRAME = [
    df1
]


TOTAL_TIMESTEPS = 1500_000
CHECKPOINT_FREQ = 100_000  # Save every N training timesteps; 0 disables snapshots.
MODEL_PREFIX = f"ppo{TOTAL_TIMESTEPS // 1000}kcnn"

N_EVAL_EPISODES = 50
STARTING_CASH = 10_000.0
EVAL_DETERMINISTIC = True

EVAL_DIR = "eval_results"

PERIODS_PER_YEAR = 252
COST_BPS = 0.0

import pandas as pd

#training seeds
START_SEED = 36
NUM_SEEDS = 10
SEEDS = list(range(START_SEED, START_SEED + NUM_SEEDS))

#configs of a and b reward values for training.py
TARGET_CONFIGS = [
    (0.75, 1.30),
    (0.75, 1.30),
    (0.50, 2.20),
    (1.25, 1.90),
]
#lookback window size for feature extractors in classes
LOOKBACKS = [5, 100]
#architrctures used in the observation space for the RL agent
ARCHITECTURE = [
    "def", "2dcnn", "gru"
]

#coefficient for the drawdown penalty in the reward calculation in training.py
FIXED_LAMBDA = 0.0

df1 = pd.read_csv("cache.csv", header=[0,1], skiprows=[2], index_col=0)
DATAFRAME = {
    "df1": df1
}

TOTAL_TIMESTEPS = 1500_000
CHECKPOINT_FREQ = 100_000  
MODEL_PREFIX = f"{TOTAL_TIMESTEPS // 1000}k"

N_EVAL_EPISODES = 50
STARTING_CASH = 10_000.0
EVAL_DETERMINISTIC = True

EVAL_DIR = "eval_results"

#trading days in a yr
PERIODS_PER_YEAR = 252

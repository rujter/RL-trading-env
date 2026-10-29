########################################
Historical trading environment
########################################

This folder is for trading on historical stock data. The agent trades one stock at a time. The idea is to try different reward settings and networks and compare the results with buy-and-hold.

It uses Gymnasium for the environment and PPO from Stable-Baselines3 for training. The code and data are in Training env.

########################################
Files
########################################

config.py - settings for the runs: seeds, reward coefficients, architectures, lookbacks, dataset and training length.
training.py - the trading environment. This handles observations, actions, fees, rewards and the training/testing ticker lists.
Classes.py - portfolio and stock classes, plus the CNN, RNN and GRU feature extractors.
run.py - runs training or evaluates saved models.
yfloader.py - downloads the historical data using yfinance.
cache.csv - the included dataset, from 2024-01-02 to 2025-12-31.
metrics.py - calculates the performance metrics for each episode.
analyse.py - takes the evaluation results and makes summary tables, heatmaps and equity curves.
MLtest.py - separate experiments with random forests, a CNN and GARCH. This is still unfinished and is not needed for PPO.

########################################
Setup
########################################

Run everything from inside historical/Training env. The paths start from this folder, so the scripts will not find the data if you run them from the repository root.

From the repository root, in PowerShell:

cd "historical/Training env"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas gymnasium stable-baselines3 torch matplotlib seaborn yfinance
.\.venv\Scripts\Activate.ps1

The commands below assume the environment is activated. Otherwise use .\.venv\Scripts\python.exe instead of python. Package versions are not fixed yet.

For MLtest.py, also install:

python -m pip install scikit-learn arch

########################################
Data
########################################

config.py loads cache.csv when it is imported:

df1 = pd.read_csv("cache.csv", header=[0, 1], skiprows=[2], index_col=0)
DATAFRAME = {"df1": df1}

The CSV has two header levels, one for the price field and one for the ticker, followed by the date-label row. The trading environment uses Close, High and Low. MLtest.py also uses Open and Volume.

To download data again:

python yfloader.py

The loader is currently set to request adjusted daily data from 2024-01-01 to 2026-01-01. It includes 30 training tickers and 14 testing tickers. Missing values are forward-filled, then any remaining gaps are filled with zero.

The loader saves cache_2024-2026.csv, but config.py looks for cache.csv. Change the filename in config.py to use the new download. There is already a cache included, so downloading again is optional.

The ticker lists are in both training.py and yfloader.py, so update both if you change them. PPO training and testing use different tickers from the same dataset. There is no automatic split by date. The start_date and end_date variables in training.py are unused and do not change which data is loaded.

########################################
Settings
########################################

Most of this is changed in config.py. Current settings:

Seeds: 36 to 45
Reward pairs (a, b): (0.75, 1.30), (0.75, 1.30), (0.50, 2.20), (1.25, 1.90)
Lookbacks: 5 and 100 bars
Architectures: def, 2dcnn, gru
Drawdown penalty (FIXED_LAMBDA): 0.0
Training steps per run: 1,500,000
Checkpoint interval: 100,000 steps
Starting cash: 10,000
Evaluation episodes per model: 50
Deterministic evaluation: True
Trading days per year for metrics: 252

def uses the default MLP features. Classes.py also supports cnn and rnn, which can be added to ARCHITECTURE. get_policy_kwargs() adjusts the network input sizes for the chosen positive integer lookback.

The RNN and GRU read the return window, then combine their output with the volatility window and the remaining features. Their hidden state is not kept between environment steps.

The first reward pair is in the list twice. With everything enabled this gives 240 runs, but only 180 different combinations. The repeated ones save to the same filenames. Start with one seed, one reward pair, one network and one lookback, with fewer training steps. Set CHECKPOINT_FREQ to zero to turn checkpoints off.

########################################
Training and testing
########################################

There are still some problems to fix before running this properly. See the notes at the end.

The switch between training and testing is tort in training.py:

tort = True   # training
tort = False  # testing

Choose one of these values, then run:

python run.py

With tort = True, the script trains a model for each combination in config.py. Training picks random training tickers and starting positions. PPO runs on the CPU, with a learning rate of 3e-4 and ent_coef=0.005. Finished models go into models/, and checkpoints go into models/checkpoints/.

With tort = False, it loads the saved models and evaluates them. Missing models are skipped without a message. This is the current default, and trained models are not included in this folder, so testing alone will not produce results until models have been trained or added.

Testing cycles through the 14 test tickers, starting from the beginning of each series and filling the lookback window before trading. With 50 episodes, some tickers are tested more than once.

After evaluation, run:

python analyse.py

You can also choose the results and plot folders:

python analyse.py --dir eval_results --out eval_plots

Keep config.py set to the runs you want to analyse. Testing skips missing models, but analyse.py stops with an error if one of its expected result files is missing.

MODEL_PREFIX comes from TOTAL_TIMESTEPS. If you change the training length, it will look for different model and result filenames too. Also, a name like df1 does not tell you which CSV was used. Keep the settings and data from each experiment so you can check this later.

########################################
How the environment works
########################################

Each episode starts with cash and no shares. The agent can buy and sell whole shares of one stock. There is no short selling or borrowing.

The observation has 2 * lookback + 4 values:

a window of daily returns;
a window of GARCH-style volatility estimates;
the high-low range divided by the closing price;
cash relative to starting cash;
the value of held shares relative to starting cash;
cash again, which is currently duplicated in the observation.

These values are scaled with tanh and returned as a flat float32 array. reset() fills the history window before the agent starts trading.

The action is one number between -1 and 1. Above 0.05 it buys, using that fraction of the available cash. Below -0.05 it sells, using the size of the action as the fraction of shares to sell. Anything in between means hold. Share quantities are rounded down to whole shares.

Orders use the next day's close with slippage added for buys and subtracted for sells. There is a comment in the code referring to the open, but the actual price comes from the close series. Slippage is calculated as:

min(0.05 * previous_volatility, 0.01)

Each executed order has a fee of 0.35% of its value, with a minimum fee of 3 cash units. The episode ends at the last bar of the series with truncated=True.

The reward is:

reward = b * portfolio_return - a * stock_return - lambda * drawdown_increase

Both returns are for the current step. The last term gives a penalty when drawdown from the highest portfolio value gets worse. This is turned off at the moment since FIXED_LAMBDA is zero.

########################################
Results
########################################

Results are saved inside Training env:

models/ - trained PPO models as .zip files.
models/checkpoints/ - checkpoints for each run.
eval_results/ - evaluation results as eval_results_<name>.npz.
eval_plots/ - summary CSVs, return and risk heatmaps, and equity curves.

The evaluation files contain price, equity, shares, order, ep_id, tickers, startingcash, deterministic and lookback. Order codes are 0 for hold, 1 for buy and 2 for sell.

The analysis calculates returns, excess return over buy-and-hold, Sharpe, Sortino, Omega, maximum drawdown, Calmar, beta, annualised alpha, exposure, turnover and buy/sell/hold counts. Results are first grouped by seed, then summarised across seeds.

analyse.py can also read random-policy results if the files are present. There is no script here that generates those files, so the random benchmark will be NaN without them.

########################################
Things that still need fixing
########################################

The filenames in run.py and analyse.py have colons, like arch:gru. Windows does not allow these in normal filenames. Change the separators in all the places that make model, checkpoint, evaluation, random-baseline and plot filenames so they still match.

In training.py, fill_price is left as None when the agent holds or cannot fill an order. float(fill_price) then causes a TypeError. It needs to return a valid price on these steps, otherwise both training and testing can fail.

The saved price array uses trade prices with slippage. Buy-and-hold is calculated from this too. When fixing the price reporting, save the market close separately for the benchmark.

(0.75, 1.30) appears twice in TARGET_CONFIGS. It repeats work and saves under the same names.

The heatmap axis labels have the roles of a and b reversed. In the reward formula, b weights portfolio return and a weights stock return.

MLtest.py still needs work. It downloads extra data for 2022-2024, prepares the data and fits GARCH models when the file runs, then calls forrestvol(). prepare_vol() tries to return inside, but the code that creates it is commented out. conv() also has problems with local variables and is set to use CUDA. It is not ready to run as a complete script yet.

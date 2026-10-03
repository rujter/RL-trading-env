from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
from collections import deque
import math

from Classes import Portfolio, Stock, PPinpoint, Order, next_order_id, nextId
SEED = 34

start_date = datetime(year=2021, month=1, day=1)
end_date = datetime(year=2025, month=1, day=1)

# set the name of the ticker we want to download market data for

tickers = [
    # Strong uptrend (winners)
    "AAPL", "MSFT", "NVDA", "COST", "V",

    # Long decline or stagnation (losers / sideways)
    "INTC", "BA", "PFE", "VZ", "T", "KHC", "CVS", "NKE", "IBM",

    # Cyclical, big swings with regime changes
    "XOM", "CVX", "F", "GM", "CAT", "FCX",

    # Defensive, low volatility
    "KO", "PG", "JNJ", "DUK",

    # High volatility
    "TSLA", "AMD",

    # Broad ETFs and other asset classes
    "SPY", "QQQ", "GLD", "TLT",
]

test_tickers = [
    # Uptrend
    "GOOGL", "HD",

    # Long decline or stagnation (the most important group)
    "DIS", "PYPL", "MMM", "CSCO",

    # Cyclical / financial
    "COP", "JPM", "DE",

    # Defensive
    "WMT", "PEP",

    # Broad ETFs and other asset classes
    "IWM", "EEM", "XLE",
]

tort = True #if true its trainig, else testing
globalctr = 0

if tort: ticker = tickers[globalctr]
else: ticker = test_tickers[random.randint(0,13)]

portfolios = {}

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




############################################CNN FOR PPO/SAC ######################################ž

class CustomCNN(BaseFeaturesExtractor):

  def __init__(
      self,
      observation_space: spaces.Box,
      features_dim: int = 128,
      price_features: int = 1,
      ind_features: int = 5,
  ):
    super().__init__(observation_space, features_dim)

    self.price_features = price_features
    self.ind_features = ind_features

    self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )

    cnn_out_dim = 32 * price_features
    total_concat_dim = cnn_out_dim + ind_features

    self.linear = nn.Sequential(
        nn.Linear(total_concat_dim, features_dim), nn.ReLU()
    )

  def forward(self, observations: th.Tensor) -> th.Tensor:
    priceobs = observations[:, : self.price_features].unsqueeze(1) 
    ind = observations[:, self.price_features :] 

    cnn_feat = self.cnn(priceobs)  
    combined = th.cat([cnn_feat, ind], dim=1) 

    return self.linear(combined)

######################################## RNN FOR PPO ####################################################3

class CustomRNN(BaseFeaturesExtractor):

    def __init__(

        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        price_features: int = 1,
        ind_features: int = 5,
        hidden_size: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.price_features = price_features
        self.ind_features = ind_features

        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)

        total_concat_dim = hidden_size + ind_features
        self.linear = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        priceobs = observations[:, : self.price_features].unsqueeze(-1)
        ind = observations[:, self.price_features :] 

        rnn_out, h_n = self.rnn(priceobs)
        last_hidden = h_n.squeeze(0)

        combined = th.cat([last_hidden, ind], dim=1) 
        return self.linear(combined)

policy_kwargs_def = None

policy_kwargs_cnn = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        price_features=10, ind_features=14, features_dim=128
    ),
)

policy_kwargs_rnn = dict(
    features_extractor_class=CustomRNN,
    features_extractor_kwargs=dict(
        price_features=10, ind_features=14, hidden_size=64, features_dim=128
    ),
)

_cache = {}

def load(t):
    if t not in _cache:
        df = yf.download(tickers=t, start=start_date, end=end_date,
                         interval="1d", group_by="ticker",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            for level in range(df.columns.nlevels):
                if set(df.columns.get_level_values(level)) == {t}:
                    df.columns = df.columns.droplevel(level)
                    break
        assert len(df) > 300, f"{t}: only {len(df)} rows downloaded"
        _cache[t] = df
    return _cache[t]

for t in tickers + test_tickers:
    load(t)
    

class Environment(gym.Env):

    def __init__(self,
                 startingcash = 5000,
                 id = 1):
        
        super().__init__()
        self.gl_counter = 0
        self.id = id
        self.startingcash = startingcash
        self.lossq = 0
        self.ticker = ticker
        self.df = load(ticker)

        portfolios[self.id] = Portfolio(id = self.id, cash = startingcash, holdings={})
        self.portfolio = portfolios[self.id]
        self.stock = Stock(id = self.ticker, pricepoint=None, name=ticker, borderArr=None, sofferArr=None)
        self.prevvolatility = 0

        self.pricesarray = deque([0,0,0,0,0,0,0,0,0,0], maxlen=10)
        self.volarray = deque([0,0,0,0,0,0,0,0,0,0], maxlen=10)

        #setting up the enviroment for the agent
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.epsilon_prev = 0

    def _get_obs(self):
        close = self.df["Close"]
        high = self.df["High"]
        low = self.df["Low"]
        i = min(self.gl_counter, len(self.df) - 1)

        price_change = 0.0 if i == 0 else (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
        self.pricesarray.append(price_change)

        if i == 0: re = 0.0
        else: re = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]

        self.lossq = self.lossq = (self.lossq * 0.5) + (min(0, re) ** 2)

        spread = (high.iloc[i] - low.iloc[i]) / close.iloc[i]

        cash_ratio = self.portfolio.cash / self.startingcash
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        holdings_ratio = (qty * close.iloc[i]) / self.startingcash

        lasttrade = close.iloc[i]
        lasttrade1 = close.iloc[i - 1] if i >= 1 else close.iloc[0]

        self.epsilon_prev = (lasttrade - lasttrade1) / lasttrade1 if lasttrade1 > 0 else 0

        volatility =  math.sqrt(0.00001+  0.07 * self.epsilon_prev**2 + 0.90 * self.prevvolatility**2)
        self.prevvolatility = volatility
        self.volarray.append(volatility)
        
        return np.array(
            list(self.pricesarray) + 
            list(self.volarray) +             [
                spread,
                cash_ratio,
                holdings_ratio,
                self.portfolio.cash / self.startingcash,
                #lossq,
            ],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        pool = tickers if tort else test_tickers
        self.ticker = random.choice(pool)
        self.df = load(self.ticker)
        self.stock = Stock(id=self.ticker, pricepoint=None, name=self.ticker,
                        borderArr=None, sofferArr=None)

        randomnum = random.randint(1, len(self.df) - 260) if tort else 1

        self.portfolio = Portfolio(id=self.id, cash=self.startingcash, holdings={})
        portfolios[self.id] = self.portfolio

        self.pricesarray.clear()
        self.pricesarray.extend([0.0] * 10)
        self.volarray.clear()
        self.volarray.extend([0.0] * 10)

        self.gl_counter = randomnum
        self.epsilon_prev = 0
        self.lossq = 0
        self.prevvolatility = 0

        return self._get_obs(), {}
    
    def _portfolio_value(self):
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        i = min(self.gl_counter, len(self.df) - 1)
        return self.portfolio.cash + qty * self.df["Close"].iloc[i]
    
    
    def step(self, action):

        mintrade = 300
        value_before = self._portfolio_value()
        stock_before = self.df["Close"].iloc[self.gl_counter]

        act = float(action[0]) if hasattr(action, "__len__") else float(action)
        order_action = "hold"

        #calculating what action the agent will take

        if act < -0.05 and self.portfolio.holdings != None and self.portfolio.holdings.get(self.stock.id, 0) > 0:
            order_action = "sell"
        elif act > 0.05 and self.portfolio.cash > 0: #dodaj tle fee
            order_action = "buy"
        else:
            order_action = "hold"
            
        #ali 3 eur ali 0.35%
        transaction_fee = 0.0035
        minfee = 3
        fee_factor = 1 + transaction_fee
        priceref = self.df["Close"].iloc[self.gl_counter]                

        num = min(1.0, abs(act))
        if order_action == "buy":

            ordersize = int((num * self.portfolio.cash) // priceref)

            if ordersize > 0:
                cost = ordersize*priceref

                if (cost * fee_factor) <= minfee: fee = minfee
                else: fee = cost*transaction_fee


                if self.portfolio.cash > ordersize*priceref+fee:
                    self.portfolio.cash -= (cost + fee)
                    self.portfolio.holdings[self.stock.id] = self.portfolio.holdings.get(self.stock.id, 0) + ordersize

            else: order_action = "hold"


        elif order_action == "sell": 

            current_holdings = self.portfolio.holdings.get(self.stock.id, 0)

            if current_holdings > 0:
                ordersize = int(num * current_holdings)

                if ordersize > 0:

                    proceeds = ordersize * priceref #torej kolko zasluzis ce prodas
                    fee = proceeds * transaction_fee if (proceeds * transaction_fee) >= 3 else 3.0

                    self.portfolio.cash += (proceeds - fee)
                    self.portfolio.holdings[self.stock.id] -= ordersize
                    if self.portfolio.holdings[self.stock.id] == 0:
                        del self.portfolio.holdings[self.stock.id]

                else: order_action = "hold"
            else: order_action = "hold"

    

        self.gl_counter += 1

        terminated = False
        truncated = self.gl_counter >= len(self.df) - 1

        # Calculate portfolio value at NEW market price
        value_after = self._portfolio_value()
        stock_after = self.df["Close"].iloc[self.gl_counter]

        port_ret = (value_after - value_before) / value_before
        stock_ret = (stock_after - stock_before) / stock_before

        downside = min(0.0, port_ret) ** 2
        lam = 3
        # Reward from portfolio change

        reward = 100*(port_ret - stock_ret) #cares about beating the marekt and downside
        #reward = ((value_after - value_before) / self.startingcash) - lam * downside #cares ab abs profit

        ordernum = 0

        if order_action == "buy":
            ordernum = 1
        elif order_action == "sell":
            ordernum = 2

        info = {
            "reward": float(reward),
            "ordernum": float(ordernum),
            "val": float(value_after),
            "price": float(priceref),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
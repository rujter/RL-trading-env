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

end_date = datetime(year=2025, month=9, day=10)
start_date = datetime(year=2025, month=1, day=1)

# set the name of the ticker we want to download market data for
ticker = "META"

# download market data for a single ticker
df_single = yf.download(
    tickers=ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    progress=False
)
df_single

if isinstance(df_single.columns, pd.MultiIndex):
    # find and drop the level whose values are just the ticker symbol
    for level in range(df_single.columns.nlevels):
        if set(df_single.columns.get_level_values(level)) == {ticker}:
            df_single.columns = df_single.columns.droplevel(level)
            break

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
      price_features: int = 10,
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
        price_features: int = 10,
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
        price_features=4, ind_features=4, features_dim=128
    ),
)

policy_kwargs_rnn = dict(
    features_extractor_class=CustomRNN,
    features_extractor_kwargs=dict(
        price_features=4, ind_features=4, hidden_size=64, features_dim=128
    ),
)

class Environment(gym.Env):

    def __init__(self,
                 startingcash = 10000,
                 id = 1):
        
        super().__init__()
        self.gl_counter = 0
        self.id = id
        self.startingcash = startingcash
        self.lossq = 0

        portfolios[self.id] = Portfolio(id = self.id, cash = startingcash, holdings={})
        self.portfolio = portfolios[self.id]
        self.stock = Stock(id = "META", pricepoint=None, name="META", borderArr=None, sofferArr=None)

        self.pricesarray = deque([0,0,0,0,0,0,0,0,0,0], maxlen=10)

        #setting up the enviroment for the agent
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        self.epsilon_prev = 0
        self.prevvolatiltiy = 0

    def _get_obs(self):
        close = df_single["Close"]
        high = df_single["High"]
        low = df_single["Low"]
        i = min(self.gl_counter, len(df_single) - 1)

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

        volatility =  math.sqrt(0.00001+  0.07 * self.epsilon_prev**2 + 0.90 * self.prevvolatiltiy**2)
        self.prevvolatiltiy = volatility
        
        return np.array(
            list(self.pricesarray) + [
                spread,
                cash_ratio,
                holdings_ratio,
                volatility,
                self.portfolio.cash / self.startingcash,
                #lossq,
            ],
            dtype=np.float32
        )

    def reset(self, seed= None, options = None):
        super().reset(seed=seed, options=options)


        self.portfolio = Portfolio(id=self.id, cash=self.startingcash)
        portfolios[self.id] = self.portfolio 

        self.price_change = 0
        self.pricesarray.clear()
        self.pricesarray.extend([0.0] * 10)

        self.gl_counter = 1 #here its 1 to test the whole thing

        self.prevvolatiltiy = 0
        self.epsilon_prev = 0
        self.lossq = 0

        return self._get_obs(), {}
    
    def _portfolio_value(self):
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        i = min(self.gl_counter, len(df_single) - 1)
        return self.portfolio.cash + qty * df_single["Close"].iloc[i]
    
    
    def step(self, action):

        mintrade = 300
        value_before = self._portfolio_value()

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
        rewarddock = 0
        priceref = df_single["Close"].iloc[self.gl_counter]                

        num = min(1.0, abs(act))
        if order_action == "buy":

            ordersize = int((num * self.portfolio.cash) // priceref)
            if ordersize <= 0: ordersize = 1

            cost = ordersize*priceref

            if (cost * fee_factor) <= minfee: fee = minfee
            else: fee = cost*transaction_fee

            #zdej pogledamo ce imamo dovolj denarja za ta order
            if self.portfolio.cash*0.2 <= (ordersize * priceref + fee):

                if self.portfolio.cash > ordersize*priceref+fee:
                    self.portfolio.cash -= (cost + fee)
                    self.portfolio.holdings[self.stock.id] = self.portfolio.holdings.get(self.stock.id, 0) + ordersize

                else: order_action = "hold"
            else: 
                order_action = "hold"

        elif order_action == "sell": 

            current_holdings = self.portfolio.holdings.get(self.stock.id, 0)

            if current_holdings > 0:
                ordersize = int(num * current_holdings)

                if ordersize <= 0: ordersize = 1

                proceeds = ordersize * priceref #torej kolko zasluzis ce prodas

                if proceeds >= mintrade:
                    # Correct conditional logic
                    fee = proceeds * transaction_fee if (proceeds * transaction_fee) >= 3 else 3.0

                    self.portfolio.cash += (proceeds - fee)
                    self.portfolio.holdings[self.stock.id] -= ordersize
                    if self.portfolio.holdings[self.stock.id] == 0:
                        del self.portfolio.holdings[self.stock.id]

                else: order_action = "hold"

        self.gl_counter += 1

        terminated = False
        truncated = self.gl_counter >= len(df_single) - 1

        # Calculate portfolio value at NEW market price
        value_after = self._portfolio_value()

        # Reward from portfolio change
        reward = ((value_after - value_before) / self.startingcash) - rewarddock

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
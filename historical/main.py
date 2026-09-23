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

end_date = datetime(year=2026, month=9, day=10)
start_date = datetime(year=2026, month=1, day=1)

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

class Environment(gym.Env):

    def __init__(self,
                 startingcash = 1000,
                 id = 1):
        
        super().__init__()
        self.gl_counter = 0
        self.id = id
        self.startingcash = startingcash

        portfolios[self.id] = Portfolio(id = self.id, cash = startingcash, holdings=None)
        self.portfolio = portfolios[self.id]
        self.stock = Stock(id = "META", pricepoint=None, name="META", borderArr=None, sofferArr=None)

        #setting up the enviroment for the agent
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        self.epsilon_prev = 0
        self.prevvolatiltiy = 0

    def _get_obs(self):
        close = df_single["Close"]
        high = df_single["High"]
        low = df_single["Low"]
        i = min(self.gl_counter, len(df_single) - 1)

        price_change = 0.0 if i == 0 else (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
        price_change2 = 0.0 if i < 2 else (close.iloc[i-1] - close.iloc[i-2]) / close.iloc[i-2]
        price_change3 = 0.0 if i < 3 else (close.iloc[i-2] - close.iloc[i-3]) / close.iloc[i-3]
        spread = (high.iloc[i] - low.iloc[i]) / close.iloc[i]

        cash_ratio = self.portfolio.cash / self.startingcash
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        holdings_ratio = (qty * close.iloc[i]) / self.startingcash
        
        self.epsilon_prev = (self.last_trade_price - self.price_before_step) / self.price_before_step
        volatility =  math.sqrt(0.00001+  0.07 * math.pow(self.epsilon_prev, 2) + 0.90 * math.pow(self.prevvolatiltiy, 2))
        self.prevvolatiltiy = volatility
        
        return np.array([
            price_change,
            price_change2,
            price_change3,
            spread,
            cash_ratio,
            holdings_ratio,
            volatility
        ], dtype=np.float32)

    def reset(self, seed= None, options = None):
        super().reset(seed=seed)

        randomnum = random.randint(1,171)

        self.portfolio = Portfolio(id=self.id, cash=self.startingcash)
        portfolios[self.id] = self.portfolio 

        self.price_change = 0
        self.price_change2 = 0
        self.price_change3 = 0

        self.gl_counter = randomnum

        self.prevvolatiltiy = 0
        self.epsilon_prev = 0

        return self._get_obs(), {}
    
    def _portfolio_value(self):
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        i = min(self.gl_counter, len(df_single) - 1)
        return self.portfolio.cash + qty * df_single["Close"].iloc[i]
    
    
    def step(self, action):
        value_before = self._portfolio_value()

        act = float(action[0]) if hasattr(action, "__len__") else float(action)
        order_action = "hold"

        #calculating what action the agent will take
        transaction_fee = 0.00005
        required_cash = df_single["Close"].iloc[self.gl_counter] * (1 + transaction_fee)


        if act < -0.05 and self.portfolio.holdings != None and self.portfolio.holdings.get(self.stock.id, 0) > 0:
            order_action = "sell"
        elif act > 0.05 and self.portfolio.cash >= required_cash:
            order_action = "buy"
        else:
            order_action = "hold"
            
        priceref = df_single["Close"].iloc[self.gl_counter]
        fee_factor = 1 + transaction_fee

        num = min(1.0, abs(act))
        if order_action == "buy":

            if self.portfolio.cash >= (priceref * fee_factor):
                usable_cash = self.portfolio.cash / fee_factor
                ordersize = int((num * usable_cash) // priceref)
                if ordersize <= 0: ordersize = 1

                cost = ordersize * priceref
                fee = cost * transaction_fee
                self.portfolio.cash -= (cost + fee)
                self.portfolio.holdings[self.stock.id] = self.portfolio.holdings.get(self.stock.id, 0) + ordersize

        elif order_action == "sell": 
            current_holdings = self.portfolio.holdings.get(self.stock.id, 0)

            if current_holdings > 0:
                priceref = df_single["Close"].iloc[self.gl_counter]                
                ordersize = int(num * current_holdings)

                if ordersize <= 0: ordersize = 1

                proceeds = ordersize * priceref
                fee = proceeds * transaction_fee
                self.portfolio.cash += (proceeds - fee)
                self.portfolio.holdings[self.stock.id] -= ordersize
                if self.portfolio.holdings[self.stock.id] == 0:
                    del self.portfolio.holdings[self.stock.id]

        self.gl_counter += 1

        terminated = False
        truncated = self.gl_counter >= len(df_single) - 1

        # Calculate portfolio value at NEW market price
        value_after = self._portfolio_value()

        # Reward from portfolio change
        reward = (value_after - value_before) / self.startingcash

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

        info = { "reward": float(reward), "ordernum": float(ordernum), "val": float(self._portfolio_value()), "price": df_single["Close"].iloc[self.gl_counter-1]}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
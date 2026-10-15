from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
from collections import deque
import math
from Classes import Portfolio, Stock, PPinpoint, Order, next_order_id, nextId
from numbers import Integral


SEED = 20
random.seed(SEED)
RETURN_SCALE = 0.02
VOLATILITY_SCALE = math.sqrt(1e-5 / (1.0 - 0.07 - 0.90))
SPREAD_SCALE = 0.03
portfolios = {}
start_date = datetime(year=2017, month=1, day=1)
end_date = datetime(year=2021, month=1, day=1)

tickers = [
    "AMZN", "META", "AVGO", "ORCL", "CRM",
    "QCOM", "ADBE", "AMAT", "MU", "GE",
    "BAC", "WFC", "MS", "GS", "BLK", "C",
    "LLY", "UNH", "ABBV", "MRK", "TMO", "AMGN",
    "MCD", "SBUX", "TGT", "LOW",
    "OXY", "SLB", "NEE", "SO",
]
test_tickers = [
    "PANW", "NOW", "INTU", "AXP", "GILD",
    "HON", "UNP", "RTX", "LMT", "EOG",
    "LIN", "NEM", "DIA", "XLK",
]


tort = False #if true its trainig, else testing


class Environment(gym.Env):

    def __init__(self,
                 #dataframe is inputted from run.py which gets it from config from DATAFRAME
                 dataframe,
                 startingcash = 5000,
                 id = 1,
                 lam = 0.2, 
                 a = 0.5,
                 b = 2,
                 lookback_window = 100):

        super().__init__()
        self.dataframe = dataframe
        if (isinstance(lookback_window, bool) or not isinstance(lookback_window, Integral)
                or lookback_window <= 0):
            raise ValueError("lookback_window must be a positive integer")
        self.lookback_window = int(lookback_window)
        self.gl_counter = 0
        self.history_steps = 0

        self.a = a
        self.b = b
        self.lam = lam

        self.test_ctr = 0   # index of the NEXT test ticker to use

        self.id = id
        self.startingcash = startingcash
        self.lossq = 0
        self.peak = self.startingcash
        self.ticker = tickers[0] if tort else test_tickers[0]

        #CSV file has 2 headers meaning its like a 2d table;
        #the first one, the first row is the close  or open
        #the second one is the ticker
        self.df = self.dataframe["Close"][self.ticker]
        self.dfl = self.dataframe["Low"][self.ticker]
        self.dfh = self.dataframe["High"][self.ticker]

        portfolios[self.id] = Portfolio(id = self.id, cash = startingcash, holdings={})
        self.portfolio = portfolios[self.id]
        self.stock = Stock(id = self.ticker, pricepoint=None, name=self.ticker, borderArr=None, sofferArr=None)
        self.prevvolatility = 0

        self.pricesarray = deque([0.0] * self.lookback_window, maxlen=self.lookback_window)
        self.volarray = deque([0.0] * self.lookback_window, maxlen=self.lookback_window)

        #setting up the enviroment for the agent
        obs_size = self.pricesarray.maxlen + self.volarray.maxlen + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.epsilon_prev = 0

    def _get_obs(self):
        
        close = self.df
        high = self.dfh
        low = self.dfl
        i = min(self.gl_counter, len(self.df) - 1)

        #adding current price to pricesarray
        current_price = close.iloc[i]
        previous_price = close.iloc[i - 1] if i > 0 else current_price
        re = (current_price - previous_price) / previous_price if previous_price > 0 else 0.0
        self.pricesarray.append(np.tanh(re / RETURN_SCALE))

        #spread calculation
        spread = (high.iloc[i] - low.iloc[i]) / current_price if current_price > 0 else 0.0
        spread_normalized = np.tanh(spread / SPREAD_SCALE)

        #normalising the cash and holdings using the tanh function
        #tanh function is used to normalise all data between [0,1]
        cash_normalized = np.tanh(self.portfolio.cash / self.startingcash)
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        holdings_normalized = np.tanh((qty * current_price) / self.startingcash)

        #calculates the volatility using garch formula
        self.epsilon_prev = re
        volatility =  math.sqrt(0.00001+  0.07 * self.epsilon_prev**2 + 0.90 * self.prevvolatility**2)
        self.prevvolatility = volatility
        self.volarray.append(np.tanh(volatility / VOLATILITY_SCALE))

        return np.array(
            list(self.pricesarray) +
            list(self.volarray) +
            [
                spread_normalized,
                cash_normalized,
                holdings_normalized,
                cash_normalized,
            ],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if tort:
            self.ticker = random.choice(tickers)
        else:
            self.ticker = test_tickers[self.test_ctr % len(test_tickers)]
            self.test_ctr += 1  

        self.df = self.dataframe["Close"][self.ticker]
        self.dfh = self.dataframe["High"][self.ticker]
        self.dfl = self.dataframe["Low"][self.ticker]
        self.stock = Stock(id=self.ticker, pricepoint=None, name=self.ticker,
                        borderArr=None, sofferArr=None)

        warmup_steps = self.pricesarray.maxlen
        if len(self.df) <= warmup_steps:
            raise ValueError(f"{self.ticker} needs at least {warmup_steps + 1} bars for warm-up and trading")
        latest_start = max(0, len(self.df) - max(260, warmup_steps + 1))
        randomnum = random.randint(0, latest_start) if tort else 0

        self.portfolio = Portfolio(id=self.id, cash=self.startingcash, holdings={})
        portfolios[self.id] = self.portfolio

        self.pricesarray.clear()
        self.pricesarray.extend([0.0] * self.pricesarray.maxlen)
        self.volarray.clear()
        self.volarray.extend([0.0] * self.volarray.maxlen)
        self.peak = self.startingcash

        self.gl_counter = randomnum
        self.history_steps = 0
        self.epsilon_prev = 0 #volatility formula use case in obs calc
        self.lossq = 0
        self.prevvolatility = 0

        # Skip unavailable leading prices before counting the warmup bars.
        while self.gl_counter < len(self.df):
            price = self.df.iloc[self.gl_counter]
            if np.isfinite(price) and price > 0:
                break
            self.gl_counter += 1
        if len(self.df) - self.gl_counter < warmup_steps + 1:
            raise ValueError(f"{self.ticker} needs {warmup_steps + 1} valid bars after its starting gap")
        for _ in range(warmup_steps):
            obs, _, _, _, _ = self.step(None)

        return obs, {}

    def _portfolio_value(self):
        qty = self.portfolio.holdings.get(self.stock.id, 0)
        i = min(self.gl_counter, len(self.df) - 1)
        return self.portfolio.cash + qty * self.df.iloc[i]


    def step(self, action):

        #warmup steps to fill the cnn and gru buffer
        if self.history_steps < self.pricesarray.maxlen:
            if self.history_steps > 0:
                self.gl_counter += 1
            obs = self._get_obs()
            self.history_steps += 1
            return obs, 0.0, False, False, {"warmup": True}

        #episode end
        if self.gl_counter >= len(self.df) - 1:
            raise RuntimeError("Episode has ended; call reset() before step().")

        value_before = self._portfolio_value()
        dd_before = (self.peak - value_before) / self.peak
        stock_before = self.df.iloc[self.gl_counter]

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
        fill_price = None

        #fills the price at the open, already decides the move beforehand
        #slippage is basedon k * prevvolatility so it doesnt fill at exactly the price you wanted
        priceref = self.df.iloc[self.gl_counter + 1]
        slip = min(0.05 * self.prevvolatility, 0.01)
        fill_buy = priceref * (1 + slip)
        fill_sell = priceref * (1 - slip)

        num = min(1.0, abs(act))
        if order_action == "buy":
            budget = num * self.portfolio.cash
            
            ordersize = int(budget // (fill_buy * fee_factor))
            while ordersize > 0:
                cost = ordersize * fill_buy
                fee = max(minfee, cost * transaction_fee)
                if cost + fee <= budget:
                    break
                ordersize -= 1

            if ordersize > 0:
                self.portfolio.cash -= (cost + fee)
                self.portfolio.holdings[self.stock.id] = self.portfolio.holdings.get(self.stock.id, 0) + ordersize
                fill_price = fill_buy
            else:
                order_action = "hold"


        elif order_action == "sell":
            current_holdings = self.portfolio.holdings.get(self.stock.id, 0)

            if current_holdings > 0:
                ordersize = int(num * current_holdings)

                if ordersize > 0:

                    proceeds = ordersize * fill_sell #torej kolko zasluzis ce prodas
                    fee = proceeds * transaction_fee if (proceeds * transaction_fee) >= 3 else 3.0

                    self.portfolio.cash += (proceeds - fee)
                    self.portfolio.holdings[self.stock.id] -= ordersize
                    if self.portfolio.holdings[self.stock.id] == 0:
                        del self.portfolio.holdings[self.stock.id]
                    fill_price = fill_sell
                else: order_action = "hold"
            else: order_action = "hold"

        self.gl_counter += 1
        terminated = False
        truncated = self.gl_counter >= len(self.df) - 1

        # Calculate portfolio value at NEW market price
        value_after = self._portfolio_value()
        stock_after = self.df.iloc[self.gl_counter]

        port_ret = (value_after - value_before) / value_before
        stock_ret = (stock_after - stock_before) / stock_before

        #drawdown calculation
        self.peak = max(self.peak, value_after)
        dd_after = (self.peak - value_after) / self.peak
        dd_increase = max(0.0, dd_after - dd_before)

        reward = self.b*port_ret - self.a * stock_ret - self.lam * dd_increase

        ordernum = 0

        if order_action == "buy":
            ordernum = 1
        elif order_action == "sell":
            ordernum = 2

        #the price we filled at 
        info = {
            "reward": float(reward),
            "ordernum": float(ordernum),
            "val": float(value_after),
            "price": float(fill_price)
        }

        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        return None

    def close(self):
        return None


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

    #TESTING TICKERSSS!!!!!!!!!
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

import pandas as pd
import yfinance as yf
from datetime import datetime

start_date = datetime(year=2021, month=1, day=1)
end_date = datetime(year=2025, month=1, day=1)

# Fetch all tickers in parallel
df_all = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

df_all.to_csv("cache.txt")

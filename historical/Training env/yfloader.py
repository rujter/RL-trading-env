from datetime import datetime
import time
import pandas as pd
import yfinance as yf

tickers = [
    #training tickers
    "AMZN", "META", "AVGO", "ORCL", "CRM",
    "QCOM", "ADBE", "AMAT", "MU", "GE",
    "BAC", "WFC", "MS", "GS", "BLK", "C",
    "LLY", "UNH", "ABBV", "MRK", "TMO", "AMGN",
    "MCD", "SBUX", "TGT", "LOW",
    "OXY", "SLB", "NEE", "SO",
    #testing tickers
    "PANW", "NOW", "INTU", "AXP", "GILD",
    "HON", "UNP", "RTX", "LMT", "EOG",
    "LIN", "NEM", "DIA", "XLK",
]

start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 1, 1)
cache_file = f"cache_{start_date.year}-{end_date.year}.csv"


def download_and_clean_data(tickers, start, end):    
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=True,
        threads=True,
        timeout=30,
    )
    df = df.sort_index(axis=1)
    df = df.ffill()
    df = df.fillna(0.0)
    return df

if __name__ == "__main__":
    df_all = download_and_clean_data(tickers, start_date, end_date)
    
    df_all.to_csv(cache_file)
    print(f"[SAVED] {cache_file} ({df_all.shape[0]} rows, {df_all.shape[1]} columns)")
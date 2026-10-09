from datetime import datetime
import time
import pandas as pd
import yfinance as yf

tickers = [
    # Training tickers (30 total)
    "AMZN", "META", "AVGO", "ORCL", "CRM",
    "QCOM", "ADBE", "AMAT", "MU", "GE",
    "BAC", "WFC", "MS", "GS", "BLK", "C",
    "LLY", "UNH", "ABBV", "MRK", "TMO", "AMGN",
    "MCD", "SBUX", "TGT", "LOW",
    "OXY", "SLB", "NEE", "SO",
    # Testing tickers (14 total)
    "PANW", "NOW", "INTU", "AXP", "GILD",
    "HON", "UNP", "RTX", "LMT", "EOG",
    "LIN", "NEM", "DIA", "XLK",
]

start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 1, 1)

cache_file = f"cache_{start_date.year}-{end_date.year}.csv"


def download_and_clean_data(tickers, start, end):
    print(f"[FETCHING] Downloading data for {len(tickers)} tickers...")
    
    # Fetch all tickers at once (yfinance handles batching internally)
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

    if df.empty:
        raise RuntimeError("Failed to fetch data from yfinance.")

    # Sort multi-index columns cleanly: (Price, Ticker)
    df = df.sort_index(axis=1)

    print("[CLEANING] Handling missing values, IPO gaps, and NaNs...")
    
    # 1. Forward fill (carries last known price forward)
    df = df.ffill()
    
    # 2. Backward fill (for stocks that hadn't had their IPO yet at start_date)
    df = df.bfill()
    
    # 3. Fill any remaining isolated NaNs with 0.0
    df = df.fillna(0.0)

    # Sanity check for remaining NaNs
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"[WARNING] {nan_count} NaNs remaining in DataFrame!")
    else:
        print("[SUCCESS] No NaNs detected in dataset.")

    return df


if __name__ == "__main__":
    df_all = download_and_clean_data(tickers, start_date, end_date)
    
    # Save clean dataset
    df_all.to_csv(cache_file)
    print(f"[SAVED] {cache_file} ({df_all.shape[0]} rows, {df_all.shape[1]} columns)")
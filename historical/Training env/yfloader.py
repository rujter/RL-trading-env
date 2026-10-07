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
    # testing tickers (14 total)
    "PANW", "NOW", "INTU", "AXP", "GILD",
    "HON", "UNP", "RTX", "LMT", "EOG",
    "LIN", "NEM", "DIA", "XLK",
]

# Change only these two lines per period
start_date = datetime(2021, 1, 1)
end_date = datetime(2026, 1, 1)

cache_file = f"cache_{start_date.year}-{end_date.year}.txt"


def fetch_chunk(chunk, tries=4):
    for i in range(tries):
        df = yf.download(
            tickers=chunk,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,   # avoids hammering Yahoo, fewer timeouts
            timeout=30,
        )
        if df is not None and not df.empty:
            return df
        wait = 5 * (i + 1)
        print(f"{chunk[0]}...: attempt {i+1} failed, retrying in {wait}s")
        time.sleep(wait)
    return None


if __name__ == "__main__":
    chunks = [tickers[i : i + 10] for i in range(0, len(tickers), 10)]
    frames = []
    for chunk in chunks:
        df = fetch_chunk(chunk)
        if df is None:
            raise RuntimeError(f"Could not fetch {chunk}")
        frames.append(df)
        time.sleep(2)

    df_all = pd.concat(frames, axis=1)

    # sanity check: any ticker that came back empty?
    got = set(df_all.columns.get_level_values(1))
    missing = [t for t in tickers if t not in got]
    if missing:
        print("[WARNING] No data for:", missing)

    df_all.to_csv(cache_file)
    print(f"[SAVED] {cache_file}  ({df_all.shape[0]} days, {len(got)} tickers)")
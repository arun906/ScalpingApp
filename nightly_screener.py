import os
from datetime import timedelta

import numpy as np
import pandas as pd

from config import (
    UNIVERSE_FILE,
    ACTIVE_WATCHLIST_FILE,
    WATCHLIST_HISTORY_FILE,
    NIFTY_SYMBOL_YF,
    MIN_PRICE,
    MAX_PRICE,
    MIN_BETA,
    MIN_AVG_VOLUME,
    NUM_LARGE_CAP,
    NUM_MID_CAP,
    get_ist_now,
)
from data_providers import YahooMarketDataProvider


def compute_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    df = pd.concat([stock_returns, index_returns], axis=1, join="inner").dropna()
    if df.empty:
        return np.nan
    s = df.iloc[:, 0]
    m = df.iloc[:, 1]
    if m.var() == 0:
        return np.nan
    cov = np.cov(s, m)[0, 1]
    var = np.var(m)
    if var == 0:
        return np.nan
    return float(cov / var)


def run_screener():
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(f"{UNIVERSE_FILE} not found. Please create it first.")

    universe_df = pd.read_csv(UNIVERSE_FILE)
    universe_df = universe_df[universe_df.get("is_active", 1) == 1].copy()

    if universe_df.empty:
        print("Universe is empty or all symbols are marked inactive.")
        return

    symbols = list(universe_df["data_symbol"].unique())
    provider = YahooMarketDataProvider()

    eod_data = provider.get_eod_history(symbols + [NIFTY_SYMBOL_YF], period="120d", interval="1d")
    nifty_df = eod_data.get(NIFTY_SYMBOL_YF)
    if nifty_df is None or nifty_df.empty:
        print("Could not fetch NIFTY data. Aborting nightly screener.")
        return

    nifty_returns = nifty_df["Close"].pct_change()

    records = []

    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        sym = row["data_symbol"]
        idx_bucket = row["index_bucket"]

        df = eod_data.get(sym)
        if df is None or df.empty:
            continue

        close_today = float(df["Close"].iloc[-1])
        if not (MIN_PRICE <= close_today <= MAX_PRICE):
            continue

        avg_volume = float(df["Volume"].tail(20).mean())
        if avg_volume < MIN_AVG_VOLUME:
            continue

        stock_returns = df["Close"].pct_change()
        beta = compute_beta(stock_returns, nifty_returns)
        if np.isnan(beta) or beta < MIN_BETA:
            continue

        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        momentum = 0.0
        if ma20 != 0 and not np.isnan(ma20):
            momentum = (close_today - ma20) / ma20

        score = beta * np.log1p(avg_volume) + 50 * momentum

        records.append(
            {
                "ticker": ticker,
                "data_symbol": sym,
                "index_bucket": idx_bucket,
                "beta": round(beta, 3),
                "avg_volume": int(avg_volume),
                "score": float(score),
            }
        )

    if not records:
        print("No stocks passed the screener filters.")
        return

    candidates_df = pd.DataFrame(records)

    large_mask = candidates_df["index_bucket"].isin(["NIFTY50", "NIFTY100"])
    mid_mask = candidates_df["index_bucket"] == "NIFTY-MIDCAP"

    large_df = candidates_df[large_mask].sort_values("score", ascending=False).head(NUM_LARGE_CAP)
    mid_df = candidates_df[mid_mask].sort_values("score", ascending=False).head(NUM_MID_CAP)

    final_df = pd.concat([large_df, mid_df], ignore_index=True)

    today = get_ist_now().date()
    tomorrow = today + timedelta(days=1)

    final_df["selection_date"] = today
    final_df["usable_for_date"] = today

    final_df.to_csv(ACTIVE_WATCHLIST_FILE, index=False)
    print(f"Saved active watchlist for {tomorrow} with {len(final_df)} stocks to {ACTIVE_WATCHLIST_FILE}")

    if os.path.exists(WATCHLIST_HISTORY_FILE):
        hist_df = pd.read_csv(WATCHLIST_HISTORY_FILE)
    else:
        hist_df = pd.DataFrame()

    hist_df = pd.concat([hist_df, final_df], ignore_index=True)

    hist_df["usable_for_date"] = pd.to_datetime(hist_df["usable_for_date"]).dt.date
    hist_df = hist_df.sort_values(["usable_for_date", "score"], ascending=False)
    hist_df = hist_df.drop_duplicates(subset=["ticker", "usable_for_date"], keep="first")

    hist_df.to_csv(WATCHLIST_HISTORY_FILE, index=False)
    print(f"Updated {WATCHLIST_HISTORY_FILE} with historical watchlist data.")


if __name__ == "__main__":
    run_screener()

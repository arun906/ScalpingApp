from datetime import date
import os

import pandas as pd

from config import ACTIVE_WATCHLIST_FILE, WATCHLIST_HISTORY_FILE, get_ist_now


def load_active_watchlist_for_date(d: date) -> pd.DataFrame:
    """Load active watchlist for given usable_for_date."""
    if not os.path.exists(ACTIVE_WATCHLIST_FILE):
        return pd.DataFrame()
    df = pd.read_csv(ACTIVE_WATCHLIST_FILE)
    if "usable_for_date" not in df.columns:
        return pd.DataFrame()
    df["usable_for_date"] = pd.to_datetime(df["usable_for_date"]).dt.date
    return df[df["usable_for_date"] == d].copy()


def load_today_watchlist() -> pd.DataFrame:
    today = get_ist_now().date()
    return load_active_watchlist_for_date(today)


def load_watchlist_history() -> pd.DataFrame:
    if not os.path.exists(WATCHLIST_HISTORY_FILE):
        return pd.DataFrame()
    df = pd.read_csv(WATCHLIST_HISTORY_FILE)
    if "usable_for_date" in df.columns:
        df["usable_for_date"] = pd.to_datetime(df["usable_for_date"]).dt.date
    if "selection_date" in df.columns:
        df["selection_date"] = pd.to_datetime(df["selection_date"]).dt.date
    return df


def save_active_watchlist(df: pd.DataFrame):
    df.to_csv(ACTIVE_WATCHLIST_FILE, index=False)


def save_watchlist_history(df: pd.DataFrame):
    df.to_csv(WATCHLIST_HISTORY_FILE, index=False)

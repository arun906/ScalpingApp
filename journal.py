import os

import pandas as pd

from config import PREDICTION_JOURNAL_FILE

PREDICTION_COLUMNS = [
    "prediction_id",
    "datetime_ist",
    "date",
    "time_bucket",
    "ticker",
    "data_symbol",
    "index_bucket",
    "nifty_trend",
    "stock_short_trend",
    "market_regime",
    "prediction_action",
    "confidence_score",
    "confidence_level_label",
    "confidence_trend",
    "confidence_volume",
    "confidence_sentiment",
    "confidence_volatility",
    "confidence_explanation",
    "valid_for_minutes",
    "price_at_prediction",
    "volume_signal",
    "sentiment_score",
    "sentiment_label",
    "news_summary",
    "news_risk_flag",
    "status_code",
    "strategy_version",
]


def ensure_prediction_journal_exists() -> pd.DataFrame:
    if os.path.exists(PREDICTION_JOURNAL_FILE):
        df = pd.read_csv(PREDICTION_JOURNAL_FILE)
    else:
        df = pd.DataFrame(columns=PREDICTION_COLUMNS)
        df.to_csv(PREDICTION_JOURNAL_FILE, index=False)
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def load_prediction_journal() -> pd.DataFrame:
    df = ensure_prediction_journal_exists()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "datetime_ist" in df.columns:
        df["datetime_ist"] = pd.to_datetime(df["datetime_ist"], errors="coerce")
    return df


def save_prediction_journal(df: pd.DataFrame):
    df.to_csv(PREDICTION_JOURNAL_FILE, index=False)


def upsert_predictions(new_preds: pd.DataFrame):
    """
    Merge new predictions into journal, updating items with same (date, time_bucket, ticker).
    """
    if new_preds.empty:
        return

    journal_df = load_prediction_journal()

    for col in PREDICTION_COLUMNS:
        if col not in journal_df.columns:
            journal_df[col] = pd.NA
        if col not in new_preds.columns:
            new_preds[col] = pd.NA

    if "date" in new_preds.columns:
        new_preds["date"] = pd.to_datetime(new_preds["date"]).dt.date

    if "datetime_ist" in new_preds.columns:
        new_preds["datetime_ist"] = pd.to_datetime(new_preds["datetime_ist"]).astype(str)

    for _, row in new_preds.iterrows():
        key_mask = (
            (journal_df["date"] == row["date"])
            & (journal_df["time_bucket"] == row["time_bucket"])
            & (journal_df["ticker"] == row["ticker"])
        )

        if key_mask.any():
            idx = journal_df[key_mask].index[0]
            for col in PREDICTION_COLUMNS:
                journal_df.at[idx, col] = row.get(col, journal_df.at[idx, col])
        else:
            journal_df = pd.concat([journal_df, pd.DataFrame([row])], ignore_index=True)

    journal_df = journal_df.drop_duplicates(subset=["date", "time_bucket", "ticker"], keep="last")
    save_prediction_journal(journal_df)

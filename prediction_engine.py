from typing import Dict
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    TZ_IST,
    get_ist_now,
    MARKET_OPEN,
    STUDY_END,
    SCALP_1_START,
    SCALP_1_END,
    NO_NEW_1_START,
    NO_NEW_1_END,
    SCALP_2_START,
    SCALP_2_END,
    NO_NEW_2_START,
    MARKET_CLOSE,
    SCALP_VALID_MINUTES,
    PREDICTION_TIME_BUCKET_MINUTES,
    MARKET_REGIME_LOOKBACK_MINUTES,
    NIFTY_SYMBOL_YF,
)
from data_providers import MarketDataProvider, NewsProvider
from sentiment_engine import simple_sentiment_from_headlines, determine_news_risk


def get_time_status(now_ist: datetime) -> Dict:
    """Return dict with code, color, human message, window_start, window_end."""
    t = now_ist.time()
    date_str = now_ist.strftime("%d-%b-%Y")
    time_str = now_ist.strftime("%I:%M %p")

    if t < MARKET_OPEN or t > MARKET_CLOSE:
        msg = f"Market is currently closed on {date_str}. Current time is {time_str} (Indian Standard Time)."
        return {"code": "CLOSED", "color": "#b22222", "message": msg, "window_start": None, "window_end": None}

    if MARKET_OPEN <= t < STUDY_END:
        msg = (
            f"Study-only period for today ({date_str}). "
            "From 9:15 AM to 10:00 AM you should only observe the market and avoid opening new scalping trades."
        )
        return {"code": "STUDY", "color": "#b22222", "message": msg, "window_start": None, "window_end": None}

    if SCALP_1_START <= t < SCALP_1_END:
        msg = (
            f"You are in Prime Scalping Window 1 on {date_str}. "
            f"Current time is {time_str}. This window runs from 10:00 AM to 11:30 AM. "
            "Short-term scalping opportunities are allowed during this period when conditions are favourable."
        )
        return {"code": "SCALP_1", "color": "#228b22", "message": msg, "window_start": None, "window_end": None}

    if NO_NEW_1_START <= t < NO_NEW_1_END:
        msg = (
            f"You are in the midday period on {date_str} (approximately 11:30 AM to 1:30 PM). "
            "This part of the session often has lower quality moves. "
            "It is generally better to avoid starting fresh scalping trades and instead manage or monitor existing positions."
        )
        return {"code": "NO_NEW_1", "color": "#ff8c00", "message": msg, "window_start": None, "window_end": None}

    if SCALP_2_START <= t < SCALP_2_END:
        msg = (
            f"You are in Prime Scalping Window 2 on {date_str}. "
            f"Current time is {time_str}. This window runs from 1:30 PM to 2:45 PM. "
            "This is another good period for short-term scalping if the market and stock conditions support it."
        )
        return {"code": "SCALP_2", "color": "#228b22", "message": msg, "window_start": None, "window_end": None}

    if NO_NEW_2_START <= t <= MARKET_CLOSE:
        msg = (
            f"You are in the final wind-down window for today ({date_str}), between approximately 2:45 PM and 3:30 PM. "
            "This period is best used for managing and exiting existing trades, not for opening new scalping positions."
        )
        return {"code": "NO_NEW_2", "color": "#ff8c00", "message": msg, "window_start": None, "window_end": None}

    msg = f"Market is open on {date_str}. Current time is {time_str}."
    return {"code": "OPEN", "color": "#1e90ff", "message": msg, "window_start": None, "window_end": None}


def floor_time_to_bucket(dt: datetime, minutes: int) -> str:
    """Floor datetime to lower time bucket of given minutes, return HH:MM string."""
    minute = (dt.minute // minutes) * minutes
    bucket_dt = dt.replace(minute=minute, second=0, microsecond=0)
    return bucket_dt.strftime("%H:%M")


def determine_stock_trend(intraday_df: pd.DataFrame, lookback_bars: int = 20) -> str:
    """Very simple intraday trend based on regression slope over recent bars."""
    if intraday_df.empty or len(intraday_df) < lookback_bars:
        return "SIDEWAYS"
    closes = intraday_df["Close"].iloc[-lookback_bars:]
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes.values, 1)[0]
    if slope > 0:
        return "UP"
    if slope < 0:
        return "DOWN"
    return "SIDEWAYS"


def compute_intraday_volatility_label(intraday_df: pd.DataFrame, lookback_bars: int = 40) -> str:
    """Rough classification of intraday volatility based on return standard deviation."""
    if intraday_df.empty or len(intraday_df) < lookback_bars:
        return "MEDIUM"  # default
    closes = intraday_df["Close"].iloc[-lookback_bars:]
    returns = closes.pct_change().dropna()
    if returns.empty:
        return "MEDIUM"
    std = returns.std()
    if std >= 0.004:  # around 0.4% or more
        return "HIGH"
    if std <= 0.0015:  # around 0.15% or less
        return "LOW"
    return "MEDIUM"


def compute_market_regime(nifty_intraday: pd.DataFrame) -> str:
    """
    Classify the overall day for NIFTY as:
    - TREND_DAY
    - RANGE_DAY
    - HIGH_VOLATILITY
    - LOW_VOLATILITY
    """
    if nifty_intraday.empty or len(nifty_intraday) < 30:
        return "UNKNOWN"

    # Focus on last MARKET_REGIME_LOOKBACK_MINUTES of data
    df = nifty_intraday.copy()
    df = df.sort_index()
    closes = df["Close"]
    if closes.empty:
        return "UNKNOWN"

    first_price = float(closes.iloc[0])
    last_price = float(closes.iloc[-1])
    day_range = float(closes.max() - closes.min())
    price_change = abs(last_price - first_price)

    if day_range == 0:
        trend_ratio = 0.0
    else:
        trend_ratio = price_change / day_range

    returns = closes.pct_change().dropna()
    intraday_vol = returns.std() if not returns.empty else 0.0

    # Classify volatility
    if intraday_vol >= 0.004:
        vol_tag = "HIGH"
    elif intraday_vol <= 0.0015:
        vol_tag = "LOW"
    else:
        vol_tag = "MEDIUM"

    # Classify trend vs range
    if trend_ratio >= 0.6:
        trend_tag = "TREND"
    else:
        trend_tag = "RANGE"

    if trend_tag == "TREND" and vol_tag in ("HIGH", "MEDIUM"):
        return "TREND_DAY"
    if trend_tag == "RANGE" and vol_tag == "HIGH":
        return "HIGH_VOLATILITY"
    if vol_tag == "LOW":
        return "LOW_VOLATILITY"
    return "RANGE_DAY"


MARKET_REGIME_DESCRIPTIONS = {
    "TREND_DAY": "The overall market (NIFTY) is showing a clear trend with reasonable intraday volatility. "
                 "Scalping in the direction of the main trend is usually favourable.",
    "RANGE_DAY": "The overall market (NIFTY) is moving sideways within a range with no clear direction. "
                 "Many scalping signals will fail or reverse quickly in this environment.",
    "HIGH_VOLATILITY": "The market is very volatile with wide swings up and down but not always a clear direction. "
                       "Opportunities exist, but risk is also higher, so trade size and discipline must be strict.",
    "LOW_VOLATILITY": "The market is relatively quiet with narrow intraday ranges. "
                      "Scalps may not move enough to justify risk, so conditions are often less attractive.",
    "UNKNOWN": "There is not enough information yet to classify the overall market regime.",
}


def generate_predictions_for_watchlist(
    watchlist_df: pd.DataFrame,
    market_data_provider: MarketDataProvider,
    news_provider: NewsProvider,
    strategy_version: str = "v1.0",
) -> pd.DataFrame:
    """
    Generate prediction rows for current time for all stocks in the active watchlist.
    Only produces actionable scalping suggestions in the two prime scalping windows.
    """
    if watchlist_df.empty:
        return pd.DataFrame()

    now_ist = get_ist_now()
    status = get_time_status(now_ist)
    status_code = status["code"]

    results = []

    # Determine NIFTY daily trend
    nifty_trend = market_data_provider.get_nifty_trend_daily()

    # Determine NIFTY intraday regime
    try:
        nifty_intraday = market_data_provider.get_intraday_history(NIFTY_SYMBOL_YF, period="1d", interval="5m")
    except Exception:
        nifty_intraday = pd.DataFrame()
    market_regime = compute_market_regime(nifty_intraday)

    # Only allow scalping signals in SCALP_1 and SCALP_2 by default
    actionable = status_code in ("SCALP_1", "SCALP_2")
    # Optional example rule: skip SCALP_2 if the day has been very low volatility
    if status_code == "SCALP_2" and market_regime == "LOW_VOLATILITY":
        actionable = False

    for _, row in watchlist_df.iterrows():
        ticker = row["ticker"]
        data_symbol = row["data_symbol"]
        index_bucket = row["index_bucket"]

        try:
            intraday = market_data_provider.get_intraday_history(data_symbol, period="1d", interval="5m")
        except Exception:
            intraday = pd.DataFrame()

        if intraday.empty:
            last_price = np.nan
            stock_trend = "SIDEWAYS"
            volume_signal = "UNKNOWN"
            volatility_label = "MEDIUM"
        else:
            intraday = intraday.sort_index()
            last_price = float(intraday["Close"].iloc[-1])
            stock_trend = determine_stock_trend(intraday, lookback_bars=20)

            recent_vol = intraday["Volume"].iloc[-20:].mean()
            hist_vol = intraday["Volume"].iloc[-60:].mean() if len(intraday) >= 60 else recent_vol
            if hist_vol == 0:
                volume_ratio = 1.0
            else:
                volume_ratio = recent_vol / hist_vol
            if volume_ratio > 1.5:
                volume_signal = "HIGH"
            elif volume_ratio < 0.7:
                volume_signal = "LOW"
            else:
                volume_signal = "NORMAL"

            volatility_label = compute_intraday_volatility_label(intraday, lookback_bars=40)

        # News, sentiment, and news risk
        headlines = news_provider.get_headlines_for_symbol(ticker, limit=5)
        sentiment_score, sentiment_label, news_summary = simple_sentiment_from_headlines(headlines)
        news_risk_flag = determine_news_risk(headlines)

        # Decide prediction_action based on trend + sentiment + market regime
        if actionable and not np.isnan(last_price):
            if nifty_trend == "BULLISH" and stock_trend == "UP" and sentiment_label != "NEGATIVE":
                prediction_action = "LONG_BIAS"
            elif nifty_trend == "BEARISH" and stock_trend == "DOWN" and sentiment_label != "POSITIVE":
                prediction_action = "SHORT_BIAS"
            else:
                prediction_action = "NO_TRADE"
        else:
            prediction_action = "NO_TRADE"

        # Confidence components
        # 1. Trend alignment component
        trend_component = 0.0
        if prediction_action == "LONG_BIAS":
            if nifty_trend == "BULLISH" and stock_trend == "UP":
                trend_component = 1.0
            elif nifty_trend in ("BULLISH", "NEUTRAL") and stock_trend == "UP":
                trend_component = 0.7
            else:
                trend_component = 0.3
        elif prediction_action == "SHORT_BIAS":
            if nifty_trend == "BEARISH" and stock_trend == "DOWN":
                trend_component = 1.0
            elif nifty_trend in ("BEARISH", "NEUTRAL") and stock_trend == "DOWN":
                trend_component = 0.7
            else:
                trend_component = 0.3

        # 2. Volume confirmation component
        if volume_signal == "HIGH":
            volume_component = 1.0
        elif volume_signal == "NORMAL":
            volume_component = 0.7
        elif volume_signal == "LOW":
            volume_component = 0.3
        else:
            volume_component = 0.5

        # 3. Sentiment support component
        if prediction_action == "LONG_BIAS":
            if sentiment_label == "POSITIVE":
                sentiment_component = 1.0
            elif sentiment_label == "NEUTRAL":
                sentiment_component = 0.7
            else:
                sentiment_component = 0.3
        elif prediction_action == "SHORT_BIAS":
            if sentiment_label == "NEGATIVE":
                sentiment_component = 1.0
            elif sentiment_label == "NEUTRAL":
                sentiment_component = 0.7
            else:
                sentiment_component = 0.3
        else:
            sentiment_component = 0.5

        # 4. Volatility suitability component (for scalping, medium-to-high is better)
        if volatility_label == "HIGH":
            volatility_component = 1.0
        elif volatility_label == "MEDIUM":
            volatility_component = 0.8
        else:  # LOW
            volatility_component = 0.4

        # Combine components into overall confidence score
        if prediction_action == "NO_TRADE" or not actionable:
            overall_confidence = 0.0
        else:
            overall_confidence = (
                0.30 * trend_component
                + 0.25 * volume_component
                + 0.20 * sentiment_component
                + 0.25 * volatility_component
            )

        # Adjust for market regime
        if market_regime == "RANGE_DAY":
            overall_confidence *= 0.7
        elif market_regime == "LOW_VOLATILITY":
            overall_confidence *= 0.6

        # Cap confidence if news risk is elevated
        if news_risk_flag in ("EVENT_RISK", "BREAKING"):
            overall_confidence = min(overall_confidence, 0.6)

        overall_confidence = float(round(overall_confidence, 2))

        # Confidence level label
        if overall_confidence >= 0.7:
            confidence_level_label = "High"
        elif overall_confidence >= 0.4:
            confidence_level_label = "Medium"
        elif overall_confidence > 0:
            confidence_level_label = "Low"
        else:
            confidence_level_label = "Very Low or No Trade"

        # Explanation text
        conf_explanation_parts = []
        if prediction_action == "LONG_BIAS":
            conf_explanation_parts.append(
                "The system currently sees this stock as a candidate for a short-term buying (long) scalp."
            )
        elif prediction_action == "SHORT_BIAS":
            conf_explanation_parts.append(
                "The system currently sees this stock as a candidate for a short-term selling or short-selling scalp."
            )
        else:
            conf_explanation_parts.append(
                "The system does not see a clear short-term scalping opportunity at this moment."
            )

        conf_explanation_parts.append(
            f"Trend alignment score is {trend_component:.2f}, volume confirmation score is {volume_component:.2f}, "
            f"news sentiment support score is {sentiment_component:.2f}, and volatility suitability score is {volatility_component:.2f}."
        )
        conf_explanation_parts.append(
            f"The overall confidence score is {overall_confidence:.2f}, classified as '{confidence_level_label}'."
        )
        if market_regime != "UNKNOWN":
            conf_explanation_parts.append(MARKET_REGIME_DESCRIPTIONS.get(market_regime, ""))
        if news_risk_flag == "EVENT_RISK":
            conf_explanation_parts.append(
                "There is an important scheduled or structural news event around this stock (for example earnings, "
                "policy decisions, corporate actions, or legal developments). Signals around such events can be more volatile."
            )
        elif news_risk_flag == "BREAKING":
            conf_explanation_parts.append(
                "There appears to be very fresh or urgent news for this stock. This can create sharp and unpredictable "
                "moves, so you should treat any signal with extra caution and possibly reduce position size."
            )

        confidence_explanation = " ".join(conf_explanation_parts)

        time_bucket = floor_time_to_bucket(now_ist, PREDICTION_TIME_BUCKET_MINUTES)
        prediction_id = f"{now_ist.strftime('%Y%m%d')}_{time_bucket}_{ticker}"

        results.append(
            {
                "prediction_id": prediction_id,
                "datetime_ist": now_ist.isoformat(),
                "date": now_ist.date(),
                "time_bucket": time_bucket,
                "ticker": ticker,
                "data_symbol": data_symbol,
                "index_bucket": index_bucket,
                "nifty_trend": nifty_trend,
                "stock_short_trend": stock_trend,
                "market_regime": market_regime,
                "prediction_action": prediction_action,
                "confidence_score": overall_confidence,
                "confidence_level_label": confidence_level_label,
                "confidence_trend": round(trend_component, 2),
                "confidence_volume": round(volume_component, 2),
                "confidence_sentiment": round(sentiment_component, 2),
                "confidence_volatility": round(volatility_component, 2),
                "confidence_explanation": confidence_explanation,
                "valid_for_minutes": SCALP_VALID_MINUTES,
                "price_at_prediction": last_price,
                "volume_signal": volume_signal,
                "sentiment_score": round(float(sentiment_score), 3),
                "sentiment_label": sentiment_label,
                "news_summary": news_summary,
                "news_risk_flag": news_risk_flag,
                "status_code": status_code,
                "strategy_version": strategy_version,
            }
        )

    return pd.DataFrame(results)

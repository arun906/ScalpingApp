import streamlit as st
import pandas as pd
from datetime import timedelta

from config import get_ist_now, PREDICTION_JOURNAL_FILE
from watchlist import load_today_watchlist, load_watchlist_history
from data_providers import YahooMarketDataProvider, StubNewsProvider, NewsAPIProvider, NEWS_API_KEY
from prediction_engine import get_time_status, generate_predictions_for_watchlist, MARKET_REGIME_DESCRIPTIONS
from journal import load_prediction_journal, upsert_predictions


def get_news_provider():
    if NEWS_API_KEY:
        return NewsAPIProvider(api_key=NEWS_API_KEY)
    return StubNewsProvider()


def map_action_to_human_text(action: str) -> str:
    if action == "LONG_BIAS":
        return "Conditions currently favour a short-term buying (long) scalping opportunity."
    if action == "SHORT_BIAS":
        return "Conditions currently favour a short-term selling or short-selling scalping opportunity."
    return "Conditions are not clear enough. The system is not recommending a new scalping trade right now."


def map_sentiment_to_text(label: str) -> str:
    if label == "POSITIVE":
        return "Overall news sentiment for this stock appears positive based on recent headlines."
    if label == "NEGATIVE":
        return "Overall news sentiment for this stock appears negative based on recent headlines."
    if label == "NEUTRAL":
        return "Overall news sentiment for this stock is neutral based on recent headlines."
    return "No clear sentiment could be derived from recent news."


def map_news_risk_to_text(flag: str) -> str:
    if flag == "NONE":
        return "No special news-related risk has been detected at this time."
    if flag == "EVENT_RISK":
        return (
            "There is an important scheduled or structural news event around this stock "
            "(for example earnings, policy decisions, corporate actions, or legal matters). "
            "Price movements may be faster and more volatile."
        )
    if flag == "BREAKING":
        return (
            "There is very fresh or breaking news around this stock. "
            "This can create sharp and unpredictable price moves. Please use extra caution."
        )
    return "News risk could not be clearly classified."


def map_status_code_to_text(code: str) -> str:
    if code == "STUDY":
        return "Study-only period. Observe the market between 9:15 AM and 10:00 AM without initiating new scalping trades."
    if code == "SCALP_1":
        return "Prime Scalping Window 1. From 10:00 AM to 11:30 AM, conditions are generally more suitable for short-term scalping."
    if code == "NO_NEW_1":
        return "Midday period. From around 11:30 AM to 1:30 PM, market quality often deteriorates, and starting new scalps is discouraged."
    if code == "SCALP_2":
        return "Prime Scalping Window 2. From 1:30 PM to 2:45 PM, there is another good opportunity window for scalping if conditions align."
    if code == "NO_NEW_2":
        return "Final wind-down period. From around 2:45 PM to 3:30 PM, focus on managing and closing existing trades rather than opening new scalps."
    if code == "CLOSED":
        return "Market is closed. No intraday trading or scalping is possible."
    return "General open-market state."


def classify_signal_freshness(age_minutes: float) -> str:
    if age_minutes <= 3:
        return "Fresh (0 to 3 minutes old) – this signal is very recent."
    if age_minutes <= 7:
        return "Aging (3 to 7 minutes old) – this signal is getting older; do not chase if price has already moved."
    return "Stale (more than 7 minutes old) – this signal is quite old, and entering now may be chasing."


def main():
    st.set_page_config(page_title="Intraday Scalping Preparation & Prediction", layout="wide")

    st.title("Intraday Scalping Preparation and Real-Time Prediction Environment")

    now_ist = get_ist_now()
    status = get_time_status(now_ist)

    # Time status banner with detailed explanation
    st.markdown(
        f"""
        <div style="padding:0.9rem; border-radius:0.6rem; background-color:{status['color']}; color:white; margin-bottom:1.2rem;">
          <b>Current Time and Trading Window Status</b><br/>
          {status['message']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Today's Live Scalping Environment",
            "Watchlists for Today and Tomorrow",
            "Historical Prediction Journal",
            "Trade Journal and Performance (Later Phase)",
        ]
    )

    market_provider = YahooMarketDataProvider()
    news_provider = get_news_provider()

    # ------------------------------------------------------------------
    # Tab 1: Today's Live Scalping Environment
    # ------------------------------------------------------------------
    with tabs[0]:
        st.header("Today’s Live Scalping Environment")

        st.write(
            "This screen shows the automatically selected watchlist for today, "
            "along with real-time scalping recommendations, explanations, news sentiment, "
            "and helpful psychological cues such as signal freshness and soft cool-down advisories."
        )

        watchlist_today = load_today_watchlist()
        if watchlist_today.empty:
            st.warning(
                "There is no active watchlist for today. "
                "The nightly screener must run before each trading day to automatically pick candidates."
            )
        else:
            st.subheader("Stocks Selected for Today’s Scalping Session (Automatically Chosen Last Night)")
            st.dataframe(
                watchlist_today[["ticker", "index_bucket", "beta", "avg_volume", "score"]],
                use_container_width=True,
            )

            if status["code"] == "CLOSED":
                st.info("Since the market is closed, live predictions are not generated at this moment.")
            else:
                st.subheader("Real-Time Scalping Suggestions and Explanations")

                new_preds = generate_predictions_for_watchlist(
                    watchlist_today, market_provider, news_provider, strategy_version="v1.0"
                )
                if not new_preds.empty:
                    upsert_predictions(new_preds)

                    journal_df = load_prediction_journal()
                    today = now_ist.date()
                    today_preds = journal_df[journal_df["date"] == today].copy()

                    if today_preds.empty:
                        st.info("No predictions have been recorded yet for today.")
                    else:
                        today_preds = today_preds.sort_values("datetime_ist").groupby("ticker").tail(1)

                        # Signal freshness
                        now = now_ist
                        today_preds["signal_age_minutes"] = (now - today_preds["datetime_ist"]).dt.total_seconds() / 60.0
                        today_preds["signal_freshness_description"] = today_preds["signal_age_minutes"].apply(
                            lambda x: classify_signal_freshness(x) if pd.notna(x) else "Age could not be calculated."
                        )

                        # Soft cool-down advisory: count signals per ticker in last 30 minutes
                        last_30_min = now - timedelta(minutes=30)
                        recent_mask = (journal_df["datetime_ist"] >= last_30_min) & (journal_df["date"] == today)
                        recent_counts = (
                            journal_df[recent_mask].groupby("ticker")["prediction_id"].count().to_dict()
                        )

                        def cooldown_message(t):
                            count = recent_counts.get(t, 0)
                            if count >= 3:
                                return (
                                    "This stock has generated several signals in the last 30 minutes. "
                                    "This is a gentle advisory to be selective and avoid over-trading on the same name."
                                )
                            if count == 2:
                                return (
                                    "This stock has produced two signals in the last 30 minutes. "
                                    "Consider whether you are reacting repeatedly to the same movement."
                                )
                            return "No soft cool-down advisory for this stock at the moment."

                        today_preds["cooldown_advisory"] = today_preds["ticker"].apply(cooldown_message)

                        # Detailed human-readable text fields
                        today_preds["scalping_recommendation_explanation"] = today_preds["prediction_action"].apply(
                            map_action_to_human_text
                        )
                        today_preds["sentiment_explanation"] = today_preds["sentiment_label"].apply(
                            map_sentiment_to_text
                        )
                        today_preds["news_risk_explanation"] = today_preds["news_risk_flag"].apply(
                            map_news_risk_to_text
                        )
                        today_preds["time_window_description"] = today_preds["status_code"].apply(
                            map_status_code_to_text
                        )
                        today_preds["market_regime_description"] = today_preds["market_regime"].apply(
                            lambda r: MARKET_REGIME_DESCRIPTIONS.get(r, "Market regime information is not available.")
                        )

                        display_df = today_preds[
                            [
                                "ticker",
                                "index_bucket",
                                "price_at_prediction",
                                "scalping_recommendation_explanation",
                                "confidence_level_label",
                                "confidence_explanation",
                                "signal_freshness_description",
                                "cooldown_advisory",
                                "sentiment_explanation",
                                "news_risk_explanation",
                                "market_regime_description",
                                "time_window_description",
                                "datetime_ist",
                            ]
                        ].copy()

                        display_df = display_df.rename(
                            columns={
                                "ticker": "Stock Symbol",
                                "index_bucket": "Index Group (NIFTY Bucket)",
                                "price_at_prediction": "Approximate Price at Time of Prediction",
                                "confidence_level_label": "Overall Confidence Level",
                                "datetime_ist": "Prediction Time (Indian Standard Time)",
                            }
                        )

                        st.dataframe(display_df, use_container_width=True)
                        st.caption(
                            f"All predictions are recorded in a structured journal file: {PREDICTION_JOURNAL_FILE}. "
                            "You can use this later for deeper analysis and model improvement."
                        )
                else:
                    st.info("The system was not able to generate any predictions at this moment. "
                            "This can happen if data is temporarily unavailable.")

    # ------------------------------------------------------------------
    # Tab 2: Watchlists for Today and Tomorrow
    # ------------------------------------------------------------------
    with tabs[1]:
        st.header("Watchlists for Today, Tomorrow, and Previous Days")

        st.write(
            "Every evening, the system runs a screener and automatically selects a filtered list of high-beta, "
            "liquid stocks from your NIFTY universe. These watchlists are stored here so you can see "
            "what was chosen for each trading day."
        )

        hist = load_watchlist_history()
        if hist.empty:
            st.info(
                "Watchlist history is empty. Please run the nightly screener at least once to populate "
                "the active watchlist and start historical logging."
            )
        else:
            unique_dates = sorted(hist["usable_for_date"].unique())
            today = get_ist_now().date()
            default_date = today if today in unique_dates else unique_dates[-1]

            selected_date = st.date_input(
                "Choose a trading date to view the watchlist that was prepared for that day:",
                value=default_date,
                min_value=min(unique_dates),
                max_value=max(unique_dates),
            )

            day_df = hist[hist["usable_for_date"] == selected_date].copy()

            if selected_date == today:
                heading = "Today’s Active Watchlist (Automatically Chosen Last Night)"
            elif selected_date > today:
                heading = "Planned Watchlist for a Future Trading Date (Preview)"
            else:
                heading = "Historical Watchlist for a Previous Trading Day"

            st.subheader(f"{heading} – Trading Date: {selected_date}")

            if day_df.empty:
                st.info("No watchlist entries were found for the selected trading date.")
            else:
                st.dataframe(
                    day_df[
                        [
                            "ticker",
                            "index_bucket",
                            "beta",
                            "avg_volume",
                            "score",
                            "selection_date",
                            "usable_for_date",
                        ]
                    ],
                    use_container_width=True,
                )

    # ------------------------------------------------------------------
    # Tab 3: Historical Prediction Journal
    # ------------------------------------------------------------------
    with tabs[2]:
        st.header("Historical Prediction Journal – All Recorded Scalping Signals")

        st.write(
            "This section allows you to review every prediction that the system has generated, "
            "including its reasoning, sentiment context, and the market regime at the time. "
            "You can filter by date, stock symbol, or recommendation type."
        )

        journal_df = load_prediction_journal()
        if journal_df.empty:
            st.info(
                "The prediction journal is currently empty. "
                "Visit the 'Today’s Live Scalping Environment' tab during market hours "
                "to generate and log predictions."
            )
        else:
            min_date = min(journal_df["date"])
            max_date = max(journal_df["date"])

            date_range = st.date_input(
                "Select the date range for which you want to review past predictions:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            if isinstance(date_range, tuple):
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date

            mask = (journal_df["date"] >= start_date) & (journal_df["date"] <= end_date)
            filtered = journal_df[mask].copy()

            tickers = sorted(filtered["ticker"].dropna().unique())
            selected_tickers = st.multiselect(
                "Filter by stock symbol:",
                options=tickers,
                default=tickers,
            )
            if selected_tickers:
                filtered = filtered[filtered["ticker"].isin(selected_tickers)]

            actions = sorted(filtered["prediction_action"].dropna().unique())
            selected_actions = st.multiselect(
                "Filter by broad recommendation type:",
                options=actions,
                default=actions,
            )
            if selected_actions:
                filtered = filtered[filtered["prediction_action"].isin(selected_actions)]

            filtered["scalping_recommendation_explanation"] = filtered["prediction_action"].apply(
                map_action_to_human_text
            )
            filtered["sentiment_explanation"] = filtered["sentiment_label"].apply(map_sentiment_to_text)
            filtered["news_risk_explanation"] = filtered["news_risk_flag"].apply(map_news_risk_to_text)
            filtered["time_window_description"] = filtered["status_code"].apply(map_status_code_to_text)
            filtered["market_regime_description"] = filtered["market_regime"].apply(
                lambda r: MARKET_REGIME_DESCRIPTIONS.get(r, "Market regime information was not available.")
            )

            display_cols = [
                "datetime_ist",
                "ticker",
                "index_bucket",
                "price_at_prediction",
                "scalping_recommendation_explanation",
                "confidence_level_label",
                "confidence_explanation",
                "sentiment_explanation",
                "news_risk_explanation",
                "market_regime_description",
                "time_window_description",
                "news_summary",
            ]

            display_df = filtered[display_cols].copy().rename(
                columns={
                    "datetime_ist": "Prediction Time (Indian Standard Time)",
                    "ticker": "Stock Symbol",
                    "index_bucket": "Index Group (NIFTY Bucket)",
                    "price_at_prediction": "Approximate Price at Time of Prediction",
                    "confidence_level_label": "Overall Confidence Level",
                    "news_summary": "Short News Headline Summary at That Time",
                }
            )

            st.dataframe(display_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 4: Trade Journal (placeholder text)
    # ------------------------------------------------------------------
    with tabs[3]:
        st.header("Trade Journal and Performance Analysis (To Be Implemented Later)")

        st.write(
            "This section is reserved for recording your own trades (entry, exit, and profit or loss) and "
            "comparing them directly against the system’s predictions. "
            "Once implemented, you will be able to see how closely you follow the plan, "
            "which windows and stocks work best for you, and where your personal discipline "
            "can be improved."
        )
        st.info(
            "For now, the prediction journal already captures all signals the system generates. "
            "You may manually maintain your own trade log or broker reports. "
            "In a later phase, that data can be imported here for automated performance analytics."
        )


if __name__ == "__main__":
    main()

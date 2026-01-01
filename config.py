import os
from datetime import time, datetime
import pytz

# Timezone
TZ_IST = pytz.timezone("Asia/Kolkata")


def get_ist_now() -> datetime:
    """Return current time in IST as timezone-aware datetime."""
    return datetime.now(TZ_IST)


# Market timings (IST)
MARKET_OPEN = time(9, 15)
STUDY_END = time(10, 0)

SCALP_1_START = time(10, 0)
SCALP_1_END = time(11, 30)

NO_NEW_1_START = SCALP_1_END
NO_NEW_1_END = time(13, 30)

SCALP_2_START = time(13, 30)
SCALP_2_END = time(14, 45)

NO_NEW_2_START = SCALP_2_END
MARKET_CLOSE = time(15, 30)

# Nightly screener config
MIN_PRICE = 100.0
MAX_PRICE = 5000.0
MIN_BETA = 1.5
MIN_AVG_VOLUME = 500000  # adjust as per liquidity preference

NUM_LARGE_CAP = 10
NUM_MID_CAP = 5

# Prediction / scalping config
PREDICTION_TIME_BUCKET_MINUTES = 15
SCALP_VALID_MINUTES = 15

# Market regime config
MARKET_REGIME_LOOKBACK_MINUTES = 120  # last 2 hours of NIFTY intraday

# Data symbols
NIFTY_SYMBOL_YF = "^NSEI"  # Yahoo Finance Nifty 50 index symbol

# File paths
UNIVERSE_FILE = "universe_master.csv"
ACTIVE_WATCHLIST_FILE = "active_watchlist.csv"
WATCHLIST_HISTORY_FILE = "watchlist_history.csv"
PREDICTION_JOURNAL_FILE = "prediction_journal.csv"
TRADE_JOURNAL_FILE = "trade_journal.csv"

# Optional: News API key from environment (if you integrate NewsAPI or similar)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

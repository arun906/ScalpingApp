from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
import requests

from config import NIFTY_SYMBOL_YF, NEWS_API_KEY


class MarketDataProvider:
    """Abstract interface for market data provider."""

    def get_eod_history(self, symbols: List[str], period: str = "120d", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def get_intraday_history(self, symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        raise NotImplementedError

    def get_last_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    def get_nifty_trend_daily(self) -> str:
        """Return simple NIFTY trend: BULLISH/BEARISH/NEUTRAL based on daily data."""
        raise NotImplementedError


class YahooMarketDataProvider(MarketDataProvider):
    """Market data provider using Yahoo Finance (free, may be delayed)."""

    def get_eod_history(self, symbols: List[str], period: str = "120d", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        data = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        result: Dict[str, pd.DataFrame] = {}
        if isinstance(data.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    df = data[sym].dropna()
                    if not df.empty:
                        result[sym] = df
                except Exception:
                    continue
        else:
            result[symbols[0]] = data.dropna()
        return result

    def get_intraday_history(self, symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        return df.dropna()

    def get_last_price(self, symbol: str) -> Optional[float]:
        df = self.get_intraday_history(symbol, period="1d", interval="1m")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])

    def get_nifty_trend_daily(self) -> str:
        """Simple NIFTY trend based on last close vs 50-day SMA."""
        data = self.get_eod_history([NIFTY_SYMBOL_YF], period="120d", interval="1d")
        df = data.get(NIFTY_SYMBOL_YF)
        if df is None or df.empty:
            return "NEUTRAL"
        closes = df["Close"]
        sma_50 = closes.rolling(window=50).mean()
        last_close = float(closes.iloc[-1])
        last_sma = float(sma_50.iloc[-1])
        if last_sma == 0 or pd.isna(last_sma):
            return "NEUTRAL"
        return "BULLISH" if last_close > last_sma else "BEARISH"


class NewsProvider:
    """Abstract interface for News provider."""

    def get_headlines_for_symbol(self, ticker: str, limit: int = 5) -> List[Dict]:
        raise NotImplementedError


class StubNewsProvider(NewsProvider):
    """
    Stub news provider that returns no headlines.
    You can later replace this with a real provider using NEWS_API_KEY.
    """

    def get_headlines_for_symbol(self, ticker: str, limit: int = 5) -> List[Dict]:
        return []


class NewsAPIProvider(NewsProvider):
    """
    Example NewsAPI.org provider.
    Requires you to set NEWS_API_KEY in environment.
    Note: Adjust query to match Indian stocks; may not perfectly map.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_headlines_for_symbol(self, ticker: str, limit: int = 5) -> List[Dict]:
        if not self.api_key:
            return []

        params = {
            "q": ticker,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
            "apiKey": self.api_key,
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            headlines: List[Dict] = []
            for a in articles[:limit]:
                headlines.append(
                    {
                        "title": a.get("title", ""),
                        "description": a.get("description", ""),
                        "published_at": a.get("publishedAt", ""),
                        "source": a.get("source", {}).get("name", ""),
                        "url": a.get("url", ""),
                    }
                )
            return headlines
        except Exception:
            return []

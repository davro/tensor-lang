"""
Market data layer.

Responsible for fetching OHLCV candles for the symbols/timeframes the bot
watches. Kept behind a small interface (MarketDataProvider) so the actual
data source can be swapped or combined without touching strategy/risk code.

GMX V2 exposes market/ticker/OHLCV data through its HTTP API
(gmxapi.io) — see https://docs.gmx.io/docs/api/overview/. Endpoint paths
and response schemas move as that API matures, so GmxRestDataProvider
below is a thin, isolated wrapper: confirm the exact path/params against
the current docs before going live, and adjust _fetch_ohlcv accordingly.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import requests
import time

TIMEFRAME_TO_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# GMX's /prices/candles endpoint only accepts these exact labels:
# 1m, 5m, 15m, 1h, 4h, 1d. Normalize common aliases so callers using
# "60m" (an hour) don't hit a 400 from the API.
GMX_PERIOD_ALIASES = {
    "60m": "1h",
    "240m": "4h",
    "1440m": "1d",
}

# GMX's candle endpoint has no weekly/monthly period at all — not an
# alias gap, the API just doesn't offer one (see TIMEFRAME_TO_SECONDS
# above). "1w"/"1M" are built here instead by resampling 1d candles into
# real calendar weeks (ISO, Monday-start) and calendar months. Values
# below are an approximate day-count per bar, used only to size how much
# daily history to request — actual bucket boundaries are real calendar
# periods (see _resample_daily_candles), so the most recent bar can be a
# few days "short" if the current week/month hasn't finished yet, same
# as any OHLCV resample of an in-progress period.
SYNTHETIC_TIMEFRAMES = {"1w": 7, "1M": 31}
MAX_SYNTHETIC_LOOKBACK_DAYS = 3650  # hard ceiling regardless of requested `limit` — ~10 years of daily candles


def _resample_daily_candles(daily_candles: List["Candle"], timeframe: str) -> List["Candle"]:
    """Aggregates 1d candles (oldest-first) into real calendar-week or
    calendar-month bars: open = first day's open in the bucket,
    close = last day's close, high/low = max/min across the bucket,
    volume = summed. Standard OHLCV resampling, just done by hand since
    this file avoids a pandas dependency for the REST path."""
    from datetime import datetime, timezone

    buckets = {}
    order = []
    for c in daily_candles:
        dt = datetime.fromtimestamp(c.timestamp, tz=timezone.utc)
        key = dt.isocalendar()[:2] if timeframe == "1w" else (dt.year, dt.month)  # (iso_year, iso_week) or (year, month)

        existing = buckets.get(key)
        if existing is None:
            buckets[key] = Candle(
                timestamp=c.timestamp, open=c.open, high=c.high,
                low=c.low, close=c.close, volume=c.volume,
            )
            order.append(key)
        else:
            existing.high = max(existing.high, c.high)
            existing.low = min(existing.low, c.low)
            existing.close = c.close
            existing.volume += c.volume

    return [buckets[k] for k in order]


@dataclass
class Candle:
    timestamp: int  # unix seconds, candle open time
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class MarketDataProvider(ABC):
    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> List[Candle]:
        """Return the most recent `limit` candles, oldest first."""
        raise NotImplementedError


class GmxRestDataProvider(MarketDataProvider):
    """
    Fetches OHLCV data from the GMX API for a given chain.

    NOTE: Confirm the live endpoint path/params against
    https://docs.gmx.io/docs/api/overview/ before relying on this in
    production — the GMX API is explicitly under active development.
    """

    def __init__(self, chain: str = "arbitrum", base_url: str = "https://arbitrum-api.gmxinfra.io"):
        self.chain = chain
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> List[Candle]:
        if timeframe in SYNTHETIC_TIMEFRAMES:
            return self._get_synthetic_candles(symbol, timeframe, limit)

        gmx_period = GMX_PERIOD_ALIASES.get(timeframe, timeframe)
        if gmx_period not in TIMEFRAME_TO_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # GMX's oracle-keeper candles endpoint. See:
        # https://docs.gmx.io/docs/api/rest-v2/
        # NOTE for TradFi symbols (GOLD, SILVER, SPCX, etc.): this
        # endpoint's tokenSymbol convention hasn't been verified against
        # those markets — it may expect a different string than the
        # market_symbol used for order placement (execution/markets.py).
        # If candles fail for a TradFi symbol, check this param first.
        # That endpoint has a ~10s route timeout on GMX's side, so
        # occasional read timeouts are expected — retry a couple times
        # with backoff before giving up on this cycle.
        params = {
            "tokenSymbol": symbol.split("/")[0],
            "period": gmx_period,
            "limit": limit,
        }

        last_exc = None
        for attempt in range(3):
            try:
                resp = self.session.get(f"{self.base_url}/prices/candles", params=params, timeout=10)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        else:
            raise last_exc

        raw = resp.json().get("candles", [])

        candles = [
            Candle(
                timestamp=int(c[0]),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]) if len(c) > 5 else 0.0,
            )
            for c in raw
        ]
        candles.sort(key=lambda c: c.timestamp)
        return candles

    def _get_synthetic_candles(self, symbol: str, timeframe: str, limit: int) -> List[Candle]:
        """Builds 1w/1M candles by resampling 1d candles — see
        SYNTHETIC_TIMEFRAMES above for why GMX can't just be asked for
        these directly. Requests enough daily history to cover roughly
        `limit` real calendar weeks/months, capped at
        MAX_SYNTHETIC_LOOKBACK_DAYS so a large `limit` (e.g. someone
        passing limit=500 for "1M") can't trigger a multi-decade daily
        candle request."""
        days_per_bar = SYNTHETIC_TIMEFRAMES[timeframe]
        days_needed = min(limit * days_per_bar + days_per_bar, MAX_SYNTHETIC_LOOKBACK_DAYS)
        daily = self.get_candles(symbol, "1d", limit=days_needed)
        resampled = _resample_daily_candles(daily, timeframe)
        return resampled[-limit:]


class CachingDataProvider(MarketDataProvider):
    """Wraps another provider with a simple in-memory cache to avoid
    hammering the API every poll cycle."""

    def __init__(self, inner: MarketDataProvider, ttl_seconds: int = 20):
        self.inner = inner
        self.ttl_seconds = ttl_seconds
        self._cache = {}

    def get_candles(self, symbol: str, timeframe: str, limit: int = 200) -> List[Candle]:
        import time

        key = (symbol, timeframe, limit)
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.ttl_seconds:
            return cached[1]

        candles = self.inner.get_candles(symbol, timeframe, limit)
        self._cache[key] = (now, candles)
        return candles
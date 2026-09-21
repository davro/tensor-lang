"""
Polls GMX's two free data surfaces and normalizes both into the same
(8,) bar-row shape tensor_store.py expects:

  - REST oracle-keeper /prices/candles  -> price (no volume)
  - GraphQL (Subsquid) tradeActions      -> volume/trade-count backfill

Deliberately two separate poll functions rather than one merged call:
candles arrive fast and cheap (15-60s cache on GMX's side), volume
backfill can lag a little behind without anything downstream breaking,
since indicators.tl already treats volume_confirmed as its own signal
of whether that data has landed yet.

This is plain Python — no TensorLang involved anywhere in this file.

Usage:
    python3 fetcher.py --market ETH --period 1h --once
    python3 fetcher.py --market ETH --period 1h --loop
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensor_store  # noqa: E402

ORACLE_BASE = "https://arbitrum-api.gmxinfra.io"
GRAPHQL_URL = "https://gmx.squids.live/gmx-synthetics-arbitrum:prod/api/graphql"

# GMX cache TTLs — no point polling faster than the source refreshes.
POLL_SECONDS = {
    "1m": 15, "5m": 15, "15m": 60, "1h": 60, "4h": 60, "1d": 60,
}

PERIOD_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
}


def fetch_candles(market: str, period: str, limit: int = 5) -> list:
    """Returns GMX's raw candle arrays: [timestamp, open, high, low, close],
    most recent first. `limit` small on purpose — we only need the
    freshest few per poll, tensor_store handles the rolling history.
    """
    resp = requests.get(
        f"{ORACLE_BASE}/prices/candles",
        params={"tokenSymbol": market, "period": period, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["candles"]


def fetch_trade_volume(market_address: str, since_ts: float) -> list:
    """Returns raw tradeActions rows newer than since_ts. Caller buckets
    these by period and merges into tensor_store separately — this
    function only talks to GMX, it doesn't know about bar periods.
    """
    query = """
    query RecentTrades($market: String!, $since: BigInt!) {
      tradeActions(
        where: { marketAddress_eq: $market, timestamp_gt: $since }
        orderBy: timestamp_ASC
        limit: 1000
      ) {
        timestamp
        sizeDeltaUsd
      }
    }
    """
    resp = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": {"market": market_address, "since": int(since_ts)}},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]["tradeActions"]


def candle_to_bar_row(candle: list) -> list:
    """GMX candle [ts, open, high, low, close] -> our 8-column row.
    Volume columns start unconfirmed; merge_volume fills them in later.
    """
    ts, o, h, l, c = candle
    return [ts, o, h, l, c, 0.0, 0.0, 0.0]


# GMX's oracle-keeper candlesticks endpoint caps at 10000 candles per
# request (docs.gmx.io/docs/api/rest-api/oracle-prices) — enough for
# most backtest/chart ranges in one call, but not necessarily "since
# this market was listed" for an old, actively-traded market. There's
# no documented pagination/cursor for going further back than that in
# one session; treat 10000 as a hard ceiling on how much history this
# tool can pull in one call, not a guarantee of full history.
GMX_CANDLE_LIMIT = 10000


def fetch_history(market: str, period: str, n_candles: int) -> np.ndarray:
    """Canonical 'get N historical bars as an (n, 8) array', oldest
    first. Shared by tools/backtest.py and tools/chart_server.py so
    there's one place that knows GMX returns candles newest-first and
    this codebase wants the opposite.
    """
    n_candles = min(n_candles, GMX_CANDLE_LIMIT)
    candles = fetch_candles(market, period, limit=n_candles)
    candles = list(reversed(candles))
    return np.array([candle_to_bar_row(c) for c in candles], dtype=np.float64)


def ensure_chart_history(market: str, period: str, n_candles: int = 5000, repo_root=None) -> np.ndarray:
    """Chart-specific cache — deliberately separate from the fixed
    200-row indicator window (tensor_store.chart_data_path vs
    data_path). Loads the cached file if one already exists; otherwise
    fetches n_candles fresh and saves it. Doesn't auto-refresh an
    existing cache — call fetch_history directly and overwrite
    chart_data_path yourself for a forced refresh (e.g. a "Refresh"
    button in chart_server.py, not built yet).
    """
    path = tensor_store.chart_data_path(market, period, repo_root)
    if path.exists():
        return np.load(path)
    bars = fetch_history(market, period, n_candles)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bars)
    return bars


def ensure_daily_reference(market: str, repo_root=None):
    """The sidebar's % change should mean the same thing for every
    market regardless of what period the chart is currently showing —
    real trading UIs always report daily change in a watchlist even
    while the chart itself is on a 5m or 1w view. This always reads
    (and, on first call for a market, fetches) '1d' data specifically,
    independent of whatever period tools/chart_server.py's dropdown is
    set to. Cheap after the first call — reuses ensure_chart_history's
    same cache file if '1d' was already viewed directly.

    Returns {"price": last_daily_close, "change_pct": ...} or None if
    there isn't at least 2 days of history yet.
    """
    bars = ensure_chart_history(market, "1d", n_candles=400, repo_root=repo_root)
    if len(bars) < 2:
        return None
    last_close = bars[-1, tensor_store.COL_CLOSE]
    prev_close = bars[-2, tensor_store.COL_CLOSE]
    if prev_close == 0:
        return None
    change_pct = (last_close - prev_close) / prev_close * 100.0
    return {"price": float(last_close), "change_pct": float(change_pct)}


def bucket_trades_by_period(trades: list, period: str) -> dict:
    """Groups raw trades into {bucket_timestamp: (volume_usd, count)}."""
    period_secs = PERIOD_SECONDS[period]
    buckets = {}
    for t in trades:
        bucket_ts = (int(t["timestamp"]) // period_secs) * period_secs
        vol, count = buckets.get(bucket_ts, (0.0, 0))
        buckets[bucket_ts] = (vol + float(t["sizeDeltaUsd"]), count + 1)
    return buckets


def fetch_markets() -> list:
    """Returns [{"symbol": ..., "address": ..., "decimals": ...}, ...]
    for every token GMX's oracle-keeper supports.

    UNVERIFIED FIELD NAMES: GMX's own docs describe this endpoint only
    as "the supported-token registry (contract address, decimals,
    synthetic flag)" — no exact JSON schema, and this couldn't be hit
    from the sandbox this was written in (arbitrum-api.gmxinfra.io
    isn't reachable from there). Parses defensively across the couple
    of shapes real GMX/GMX-adjacent APIs commonly use. If your real
    response doesn't match, `print(resp.json())` once to see the
    actual shape — this is a five-minute field-name fix, not a
    redesign.
    """
    resp = requests.get(f"{ORACLE_BASE}/tokens", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    raw = data.get("tokens", data) if isinstance(data, dict) else data

    markets = []
    if isinstance(raw, list):
        for t in raw:
            markets.append({
                "symbol": t.get("symbol") or t.get("tokenSymbol"),
                "address": t.get("address") or t.get("tokenAddress"),
                "decimals": t.get("decimals"),
            })
    elif isinstance(raw, dict):
        for symbol, info in raw.items():
            markets.append({
                "symbol": info.get("symbol", symbol) if isinstance(info, dict) else symbol,
                "address": info.get("address") if isinstance(info, dict) else None,
                "decimals": info.get("decimals") if isinstance(info, dict) else None,
            })
    return [m for m in markets if m["symbol"]]


def backfill(market: str, period: str, market_address: str = None) -> None:
    """Seeds a full (200, 8) window in one request instead of waiting for
    ~200 poll cycles to fill it live. Overwrites data_path outright
    (not upsert_bar in a loop — 200 individual np.save calls would be
    both slow and pointless when we can just build the array directly).
    If GMX returns fewer than WINDOW_LEN candles (a young market), the
    remaining rows stay zero-padded at the front, same convention as a
    freshly-initialized window.
    """
    candles = fetch_candles(market, period, limit=tensor_store.WINDOW_LEN)
    candles = list(reversed(candles))  # GMX returns newest-first; we want oldest-first
    rows = [candle_to_bar_row(c) for c in candles]
    n = len(rows)

    window = np.zeros((tensor_store.WINDOW_LEN, tensor_store.N_FEATURES), dtype=np.float64)
    if n > 0:
        window[tensor_store.WINDOW_LEN - n:, :] = np.array(rows, dtype=np.float64)

    path = tensor_store.data_path(market, period)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, window)

    if market_address and n > 0:
        earliest_ts = float(window[tensor_store.WINDOW_LEN - n, tensor_store.COL_TIMESTAMP])
        trades = fetch_trade_volume(market_address, earliest_ts)
        for bucket_ts, (vol, count) in bucket_trades_by_period(trades, period).items():
            tensor_store.merge_volume(market, period, bucket_ts, vol, count)

    tensor_store.write_cache_window(market, period)


def poll_once(market: str, period: str, market_address: str = None) -> None:
    candles = fetch_candles(market, period)
    for candle in reversed(candles):  # oldest first, so ordering into the window is correct
        tensor_store.upsert_bar(market, period, candle_to_bar_row(candle))

    if market_address:
        since = tensor_store.last_confirmed_timestamp(market, period)
        trades = fetch_trade_volume(market_address, since)
        for bucket_ts, (vol, count) in bucket_trades_by_period(trades, period).items():
            tensor_store.merge_volume(market, period, bucket_ts, vol, count)

    tensor_store.write_cache_window(market, period)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, help="e.g. ETH, BTC, XAU")
    parser.add_argument("--period", required=True, choices=list(PERIOD_SECONDS))
    parser.add_argument("--market-address", default=None, help="GMX market contract address, for volume backfill")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="poll a single time, then exit (default)")
    group.add_argument("--loop", action="store_true", help="poll forever instead of once")
    group.add_argument("--backfill", action="store_true",
                        help="seed a full window in one request instead of trickling in live")
    args = parser.parse_args()

    if args.backfill:
        backfill(args.market, args.period, args.market_address)
        print(f"Backfilled: {args.market} {args.period}")
        return

    if not args.loop:
        poll_once(args.market, args.period, args.market_address)
        print(f"Polled once: {args.market} {args.period}")
        return

    interval = POLL_SECONDS[args.period]
    print(f"Polling {args.market} {args.period} every {interval}s (Ctrl+C to stop)")
    while True:
        try:
            poll_once(args.market, args.period, args.market_address)
        except requests.RequestException as e:
            print(f"[fetcher] request failed, will retry next tick: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()

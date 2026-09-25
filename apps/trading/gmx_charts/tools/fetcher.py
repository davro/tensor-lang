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
import json
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
    fetches n_candles fresh and saves it.

    NOTE: this never refreshes an existing cache file — once
    data/{market}_{period}_chart.npy exists, this returns it forever,
    however old. Kept around only because ensure_chart_history_fresh
    (below) falls back to fetching via the same path when there's no
    cache yet; nothing in chart_server.py should call this one
    directly for display anymore. See ensure_chart_history_fresh for
    the version that actually keeps the chart current.
    """
    path = tensor_store.chart_data_path(market, period, repo_root)
    if path.exists():
        return np.load(path)
    bars = fetch_history(market, period, n_candles)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bars)
    return bars


def _merge_tail(bars: np.ndarray, tail_rows: list) -> np.ndarray:
    """Merges a small batch of freshly-fetched rows (newest few candles,
    oldest-first) into an existing bars array: updates a row in place if
    its timestamp already exists (the still-forming current candle, or a
    late correction to a recent one), appends it if newer than anything
    on file, ignores anything older than the array's earliest bar.
    Keeps bars sorted oldest-first throughout, same convention as
    fetch_history."""
    if len(bars) == 0:
        return np.array(tail_rows, dtype=np.float64)

    timestamps = bars[:, tensor_store.COL_TIMESTAMP]
    for row in tail_rows:
        ts = row[tensor_store.COL_TIMESTAMP]
        if ts < timestamps[0]:
            continue  # older than our whole window — irrelevant
        if ts > timestamps[-1]:
            bars = np.vstack([bars, row])
            timestamps = bars[:, tensor_store.COL_TIMESTAMP]
            continue
        idx = int(np.searchsorted(timestamps, ts))
        if idx < len(bars) and timestamps[idx] == ts:
            bars[idx] = row  # exact match — update in place (still-forming or corrected candle)
    return bars


def ensure_chart_history_fresh(
    market: str, period: str, n_candles: int = 5000, repo_root=None, max_age_seconds: int = None,
) -> np.ndarray:
    """Like ensure_chart_history, but actually keeps the data current:
    if the cache file is older than max_age_seconds (defaults to that
    period's own POLL_SECONDS — no point calling a 1h chart 'stale'
    every 5 seconds when GMX itself only refreshes it every 60s), this
    re-fetches just the last few candles (cheap — same small `limit` as
    fetcher.py's own poll_once) and merges them in, rather than
    re-downloading the full n_candles history on every click.

    On a fetch failure (network hiccup, GMX rate limit) this logs
    nothing and just returns the existing — possibly still-stale —
    cache rather than raising, so a transient error doesn't blank the
    chart the person is currently looking at; the next click or the
    background refresher (see chart_server.py) gets another chance.
    """
    path = tensor_store.chart_data_path(market, period, repo_root)
    max_age = max_age_seconds if max_age_seconds is not None else POLL_SECONDS.get(period, 60)

    if not path.exists():
        bars = fetch_history(market, period, n_candles)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, bars)
        return bars

    bars = np.load(path)
    age = time.time() - path.stat().st_mtime
    if age < max_age:
        return bars

    try:
        tail_candles = fetch_candles(market, period, limit=5)
        tail_rows = [candle_to_bar_row(c) for c in reversed(tail_candles)]
        bars = _merge_tail(bars, tail_rows)
        np.save(path, bars)
    except requests.RequestException:
        pass  # keep serving the stale cache rather than erroring the chart out

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
    bars = ensure_chart_history_fresh(market, "1d", n_candles=400, repo_root=repo_root)
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

    NOTE: /markets (see fetch_market_backing below) turned out to BE
    reachable and has a confirmed, verified schema — if this function's
    field-name guessing ever causes trouble, /markets' "name" field
    ("SOL/USD [SOL-USDC]") is a viable alternate source for the base
    symbol list, not just for backing classification.
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


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def fetch_market_backing() -> dict:
    """Classifies each base symbol as spot-backed ("buyable" — GMX
    actually holds a real asset genuinely pegged to it in the market's
    pool) vs synthetic-only (perp price exposure only — the pool
    backing it has no real relationship to the traded asset at all,
    e.g. GOLD/USD is backed by ETH+USDC, not gold).

    CONFIRMED against the live endpoint: GET /markets returns a list of
    {"name": "SOL/USD [SOL-USDC]", "indexToken": "0x2bcC...",
    "longToken": "0x2bcC...", "shortToken": "0xaf88...", "isListed":
    true, ...} per market.

    Classification is based on the "name" field's own [pool] bracket,
    NOT on comparing indexToken's address to longToken/shortToken's
    address directly — that seemed like the obvious approach but is
    WRONG for at least BTC: every BTC/USD market's indexToken is one
    fixed oracle price-identifier address that doesn't literally equal
    the ERC20 address of ANY of the real wrapped-BTC tokens that back
    it (WBTC.b, tBTC) — GMX evidently tracks "the BTC price" as one
    abstract identifier separate from whichever specific BTC-pegged
    token happens to be pool collateral, so raw address equality
    misclassifies genuinely spot-backed BTC pools as synthetic. GMX's
    own display name doesn't have that problem: a symbol is treated as
    buyable if the base symbol appears (case-insensitively, as a
    substring — covers wrapped/staked variants like "WBTC.b" or
    "wstETH") in either side of that market's own [long-short] pool
    label. "BTC" appearing inside "WBTC.b" is exactly the signal we
    want; two BTC markets backed by unrelated collateral (tBTC vs
    USDG) get classified independently and correctly either way.

    A single symbol can have SEVERAL markets/pools on GMX (SOL/USD
    alone has SOL-USDC, WBTC.b-USDC, and USDG-USDG pools) — this treats
    a symbol as buyable if ANY of its listed markets is spot-backed,
    since that's enough for the real asset to be genuinely held
    somewhere in GMX's liquidity.

    Skips: unlisted markets (isListed: false — deprecated tickers) and
    SWAP-ONLY entries (pure swap pools, no price feed, not a tradable
    symbol at all — identifiable by the "SWAP-ONLY" name prefix or an
    all-zero indexToken).

    Returns {symbol: True/False}. Symbols that only ever appear as
    SWAP-ONLY/unlisted never appear in the result at all — callers
    should treat a missing key as "unknown", not "synthetic".
    """
    resp = requests.get(f"{ORACLE_BASE}/markets", timeout=10)
    resp.raise_for_status()
    markets = resp.json().get("markets", [])

    buyable_symbols = set()
    all_symbols = set()

    for m in markets:
        if not m.get("isListed", True):
            continue
        index_token = (m.get("indexToken") or "").lower()
        name = m.get("name", "")
        if index_token == ZERO_ADDRESS or name.startswith("SWAP-ONLY"):
            continue  # pure swap pool — no price feed, not a symbol

        if "/" not in name or "[" not in name or "]" not in name:
            continue
        base_symbol = name.split("/", 1)[0].strip()
        if not base_symbol:
            continue
        all_symbols.add(base_symbol)

        pool_label = name[name.find("[") + 1: name.find("]")]
        pool_tokens = [t.strip().lower() for t in pool_label.split("-")]
        if any(base_symbol.lower() in tok for tok in pool_tokens):
            buyable_symbols.add(base_symbol)

    return {symbol: (symbol in buyable_symbols) for symbol in all_symbols}


def market_backing_cache_path(repo_root=None) -> Path:
    return tensor_store.chart_data_path("_PLACEHOLDER_", "1d", repo_root).parent / "market_backing.json"


def ensure_market_backing(max_age_days: int = 7, repo_root=None) -> dict:
    """Loads the cached buyable/synthetic classification, refetching if
    it's missing or older than max_age_days. Market backing composition
    (which pools exist, what backs them) changes far slower than price
    data — new markets get listed occasionally, existing ones almost
    never change what backs them — so a multi-day cache is appropriate
    here in a way it wouldn't be for candle data. Returns {} (meaning
    "unknown for everything") on a fetch failure rather than raising,
    so a GMX hiccup degrades to "no backing badges shown" instead of
    crashing the chart server."""
    path = market_backing_cache_path(repo_root)
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass  # fall through and refetch

    try:
        backing = fetch_market_backing()
    except requests.RequestException:
        return {}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(backing))
    return backing


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
"""
Rolling (200, 8) bar-window store for apps/trading/gmx_charts.

Deliberately plain numpy, not TensorLang: window shifting (drop oldest
row, append newest) is data plumbing, not differentiable computation.
apps/games/tic_tac_toe/infer.tl's own comment makes the same call for
board-cell masking — "this keeps the network itself simple" — so we
follow that precedent here rather than doing it in a .tl file.

Column layout (must match indicators.tl's comment block exactly):
    0 timestamp | 1 open | 2 high | 3 low | 4 close
    5 volume_usd | 6 volume_confirmed | 7 trade_count

Two locations, two lifetimes:
  - data/{market}_{period}.npy   durable, one file per market+period,
                                  grows/rolls forever across restarts.
  - cache/apps/trading/gmx_charts/indicators.tl/window.npy
                                  ephemeral single-slot input, written
                                  fresh right before every indicators.tl
                                  invocation (mirrors board.npy in
                                  apps/games/tic_tac_toe/tools/agent.py).
"""
import sys
from pathlib import Path

import numpy as np

# from apps/trading/gmx_charts/tools/tensor_store.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

WINDOW_LEN = 200
N_FEATURES = 8
APP = "apps/trading/gmx_charts"

# Column indices, exported so callers don't hardcode magic numbers.
COL_TIMESTAMP = 0
COL_OPEN = 1
COL_HIGH = 2
COL_LOW = 3
COL_CLOSE = 4
COL_VOLUME_USD = 5
COL_VOLUME_CONFIRMED = 6
COL_TRADE_COUNT = 7


def data_path(market: str, period: str, repo_root: Path = None) -> Path:
    repo_root = repo_root or chunked_runner.find_repo_root()
    return repo_root / "apps" / "trading" / "gmx_charts" / "data" / f"{market}_{period}.npy"


def chart_data_path(market: str, period: str, repo_root: Path = None) -> Path:
    """Separate from data_path(): the indicator window is deliberately
    fixed at WINDOW_LEN=200 rows because indicators.tl needs a fixed-
    shape tensor. A chart has no such constraint and should show as
    much history as the person wants to scroll through — reusing the
    200-row file capped every chart at ~8 days on a 1h period, which
    is a Chart Renderer bug, not a GMX data limit.
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    return repo_root / "apps" / "trading" / "gmx_charts" / "data" / f"{market}_{period}_chart.npy"


def load_or_init(market: str, period: str, repo_root: Path = None) -> np.ndarray:
    """Loads the durable window, or returns a zero-filled (200, 8) array
    if this market+period has never been seen before. Callers should
    treat an all-zero row as 'not populated yet', same spirit as
    indicators.tl's volume_confirmed flag.
    """
    path = data_path(market, period, repo_root)
    if path.exists():
        return np.load(path)
    return np.zeros((WINDOW_LEN, N_FEATURES), dtype=np.float64)


def upsert_bar(market: str, period: str, bar_row: np.ndarray, repo_root: Path = None) -> np.ndarray:
    """bar_row: shape (8,) or (1, 8), in the column layout above.

    If bar_row's timestamp matches the current last row's timestamp,
    this is a still-forming candle — overwrite in place. Otherwise it's
    a newly closed candle — shift the window and append. Returns the
    updated (200, 8) array and persists it to data_path.
    """
    bar_row = np.asarray(bar_row, dtype=np.float64).reshape(1, N_FEATURES)
    window = load_or_init(market, period, repo_root)

    last_ts = window[-1, COL_TIMESTAMP]
    new_ts = bar_row[0, COL_TIMESTAMP]

    if last_ts == new_ts:
        window = window.copy()
        window[-1, :] = bar_row[0, :]
    else:
        window = np.vstack([window[1:, :], bar_row])

    path = data_path(market, period, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, window)
    return window


def merge_volume(market: str, period: str, bucket_ts: float, volume_usd: float,
                  trade_count: int, repo_root: Path = None) -> np.ndarray:
    """Finding 'the row where timestamp == bucket_ts' is a value search,
    which TensorLang can't do (no dynamic indexing) — this stays host-
    side by design, not as a workaround.
    """
    window = load_or_init(market, period, repo_root)
    matches = np.where(window[:, COL_TIMESTAMP] == bucket_ts)[0]
    if len(matches) == 0:
        # Volume arrived before the candle did (GraphQL lagging REST) —
        # drop it silently for now rather than guessing which row it
        # belongs to. Revisit if this turns out to happen often.
        return window
    row = matches[-1]
    window = window.copy()
    window[row, COL_VOLUME_USD] = volume_usd
    window[row, COL_VOLUME_CONFIRMED] = 1.0
    window[row, COL_TRADE_COUNT] = trade_count

    path = data_path(market, period, repo_root)
    np.save(path, window)
    return window


def write_cache_window(market: str, period: str, entry_file: str = "indicators.tl",
                        repo_root: Path = None) -> Path:
    """Copies the durable window into the given entry .tl file's fixed
    input slot, creating the cache directory if this is the first run.
    entry_file lets tools/backtest.py A/B indicators.tl against
    indicators_no_gate.tl without them clobbering each other's cache.
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    window = load_or_init(market, period, repo_root)
    return write_cache_window_from_array(window, entry_file, repo_root)


def write_cache_window_from_array(window: np.ndarray, entry_file: str = "indicators.tl",
                                   repo_root: Path = None) -> Path:
    """Same destination as write_cache_window, but takes an explicit
    (200, 8) array instead of reading the durable data_path file.
    Used by tools/backtest.py, which slides a window across historical
    bars that were never written to data/ at all — going through
    write_cache_window there would silently score against the live
    store instead of the historical slice being tested.
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    cache_dir = repo_root / "cache" / "apps" / "trading" / "gmx_charts" / entry_file
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "window.npy"
    np.save(out_path, window)
    return out_path


def last_confirmed_timestamp(market: str, period: str, repo_root: Path = None) -> float:
    """Cursor for the Fetcher's GraphQL trade-volume backfill, so it
    doesn't re-query from genesis on every poll.
    """
    window = load_or_init(market, period, repo_root)
    confirmed = window[window[:, COL_VOLUME_CONFIRMED] == 1.0]
    if len(confirmed) == 0:
        return 0.0
    return float(confirmed[-1, COL_TIMESTAMP])

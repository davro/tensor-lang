"""
Slides a 200-row window across historical GMX candles, calling
indicators.tl once per window position, and scores each signal against
what actually happened in the following bar.

Cost warning, read before running a large range: each window position
is a full separate tensorlang.py subprocess (compile + CUDA kernel
launch) — the same cost as one live tick in signal_agent.py, not a
cheap in-process function call. --stride and --max-windows exist so
you can time a small run first rather than discover the real cost 500
windows in.

Statistical caveats, on purpose, not swept under the rug:
  - Overlapping windows are NOT independent samples — consecutive
    windows share 199 of 200 rows, so "100 signals" is nowhere near
    100 independent trials. Treat the win rate as a rough signal, not
    a p-value.
  - No transaction costs, slippage, or funding fees are modeled here.
    GMX's own funding/borrowing costs alone could flip a marginal
    result.
  - This is one asset (ETH), one timeframe (whatever --period you
    pass), one parameter set (SMA 10/30, RSI 14, thresholds 30/70).
    A result here says nothing about other markets or other
    parameters without re-running against them explicitly.

Usage:
    python3 backtest.py --market ETH --period 1h --candles 2000 --stride 24 --max-windows 50
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetcher  # noqa: E402
import tensor_store  # noqa: E402
import signal_agent  # noqa: E402



def load_history(market: str, period: str, n_candles: int, snapshot_path: str = None) -> np.ndarray:
    """Loads historical bars either fresh from GMX or from a saved
    snapshot. GMX's candles endpoint only supports "the most recent N
    candles as of right now" — there's no documented from/to timestamp
    parameter — so re-running a fresh fetch later tests a DIFFERENT,
    forward-shifted window, not the same one. Snapshotting sidesteps
    that entirely rather than relying on undocumented API behavior:
    once saved, every comparison run (e.g. --entry indicators.tl vs
    --entry indicators_no_gate.tl) sees byte-identical history.
    """
    if snapshot_path and Path(snapshot_path).exists():
        print(f"Loading snapshot: {snapshot_path}")
        return np.load(snapshot_path)

    bars = fetcher.fetch_history(market, period, n_candles)
    if snapshot_path:
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(snapshot_path, bars)
        print(f"Saved snapshot: {snapshot_path}")
    return bars


def run_backtest(market: str, period: str, n_candles: int, stride: int,
                  max_windows: int, repo_root=None, entry_file: str = "indicators.tl",
                  snapshot_path: str = None) -> list:
    bars = load_history(market, period, n_candles, snapshot_path)
    window_len = tensor_store.WINDOW_LEN

    if len(bars) < window_len + 1:
        raise RuntimeError(
            f"Only {len(bars)} candles available for {market} {period}, "
            f"need at least {window_len + 1} (window + one bar to score against)"
        )

    # end_idx is the index of the most recent bar IN the window (i.e. the
    # decision point); bars[end_idx] is the next bar we score against.
    end_indices = list(range(window_len, len(bars) - 1, stride))
    if max_windows:
        end_indices = end_indices[:max_windows]

    results = []
    for i, end_idx in enumerate(end_indices):
        window = bars[end_idx - window_len:end_idx]
        tensor_store.write_cache_window_from_array(window, entry_file, repo_root)

        t0 = time.time()
        try:
            raw = signal_agent.run_inference(market, period, repo_root, skip_cache_write=True,
                                              entry_file=entry_file)
            error = None
        except RuntimeError as e:
            raw, error = None, str(e)
        elapsed = time.time() - t0

        entry_close = bars[end_idx - 1, tensor_store.COL_CLOSE]
        next_close = bars[end_idx, tensor_store.COL_CLOSE]
        pct_return = (next_close - entry_close) / entry_close

        action = "HOLD" if raw is None else ("BUY" if raw > 0 else "SELL" if raw < 0 else "HOLD")
        results.append({
            "timestamp": bars[end_idx - 1, tensor_store.COL_TIMESTAMP],
            "action": action,
            "raw": raw,
            "next_period_return": pct_return,
            "error": error,
        })
        status = error if error else f"next-bar return {pct_return:+.4%}"
        print(f"[{i + 1}/{len(end_indices)}] {action:4s}  {status}  ({elapsed:.2f}s)")

    return results, bars, end_indices


def scorecard(results: list, bars: np.ndarray, end_indices: list) -> None:
    signals = [r for r in results if r["action"] != "HOLD" and r["error"] is None]

    # Buy-and-hold over the SAME span the signals were actually scored
    # against — entry of the first window to the next-bar close of the
    # last window. Comparing to the full fetched history instead (an
    # earlier bug in this script) overstates or understates the
    # benchmark depending on --candles vs --stride/--max-windows.
    range_start_idx = end_indices[0] - 1
    range_end_idx = end_indices[-1]
    range_start_price = bars[range_start_idx, tensor_store.COL_CLOSE]
    range_end_price = bars[range_end_idx, tensor_store.COL_CLOSE]
    buy_and_hold_return = (range_end_price - range_start_price) / range_start_price

    print("\n--- Scorecard ---")
    print(f"windows evaluated:              {len(results)}")
    print(f"non-HOLD signals:               {len(signals)}")
    print(f"buy-and-hold over SAME range:   {buy_and_hold_return:+.4%}")

    if not signals:
        print("No BUY/SELL signals fired in this range — nothing further to score.")
        return

    hits = sum(
        1 for r in signals
        if (r["action"] == "BUY" and r["next_period_return"] > 0)
        or (r["action"] == "SELL" and r["next_period_return"] < 0)
    )
    avg_signal_return = sum(
        r["next_period_return"] if r["action"] == "BUY" else -r["next_period_return"]
        for r in signals
    ) / len(signals)

    print(f"win rate:                     {hits / len(signals):.1%}")
    print(f"avg next-bar return / signal: {avg_signal_return:+.4%}")
    print(
        "\nReminder: overlapping windows aren't independent samples, and "
        "this ignores fees/slippage/funding entirely — treat this as a "
        "rough first read, not a verdict."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--candles", type=int, default=2000,
                         help="how much history to pull for the backtest range (max %d)" % fetcher.GMX_CANDLE_LIMIT)
    parser.add_argument("--stride", type=int, default=24,
                         help="bars to advance between window positions (default: 24, e.g. once/day at 1h period)")
    parser.add_argument("--max-windows", type=int, default=50,
                         help="cap on subprocess invocations for a first, cheap timing run (default: 50)")
    parser.add_argument("--entry", default="indicators.tl",
                         help="which .tl script to run, e.g. indicators_no_gate.tl for the RSI-gate A/B")
    parser.add_argument("--save-snapshot", default=None,
                         help="path to save the fetched history to, for reuse in later comparison runs")
    parser.add_argument("--load-snapshot", default=None,
                         help="path to a previously saved snapshot — reuses it instead of fetching fresh, "
                              "so --entry indicators.tl and --entry indicators_no_gate.tl runs are compared "
                              "on byte-identical history")
    args = parser.parse_args()

    snapshot_path = args.load_snapshot or args.save_snapshot
    results, bars, end_indices = run_backtest(
        args.market, args.period, args.candles, args.stride, args.max_windows,
        entry_file=args.entry, snapshot_path=snapshot_path,
    )
    scorecard(results, bars, end_indices)


if __name__ == "__main__":
    main()

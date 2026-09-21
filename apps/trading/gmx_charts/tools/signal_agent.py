"""
Wraps apps/trading/gmx_charts/indicators.tl behind a plain
`get_signal(market, period)` function, the same way
apps/games/tic_tac_toe/tools/agent.py wraps infer.tl — callers don't
need to know about subprocesses, .npy files, or TensorLang.

One deliberate difference from agent.py's pattern: tic_tac_toe falls
back to a random legal move if the TensorLang subprocess fails, since
a random move keeps a game playable. There is no equivalent safe
default for a trading decision — a random buy/sell would be a real
capital risk — so this reports HOLD on any failure instead, and always
surfaces the failure reason rather than swallowing it silently.
"""
import sys
from pathlib import Path

import numpy as np

# from apps/trading/gmx_charts/tools/signal_agent.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensor_store  # noqa: E402

APP = "apps/trading/gmx_charts"


def run_inference(market: str, period: str, repo_root=None, skip_cache_write: bool = False,
                   entry_file: str = "indicators.tl") -> float:
    """Writes the current window, runs entry_file as a subprocess,
    returns the raw composite float (see indicators.tl for encoding).
    Raises RuntimeError on failure — callers decide how to degrade.

    skip_cache_write=True: assume the caller (tools/backtest.py) already
    wrote the exact window it wants tested to the cache slot itself —
    don't overwrite it from the durable live store.

    entry_file: which .tl script to run — indicators.tl by default, or
    indicators_no_gate.tl for the RSI-gate A/B comparison.
    """
    import subprocess

    repo_root = repo_root or chunked_runner.find_repo_root()
    if not skip_cache_write:
        tensor_store.write_cache_window(market, period, entry_file, repo_root)

    result = subprocess.run(
        [sys.executable, "tensorlang.py", f"{APP}/{entry_file}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    signal_path = repo_root / "cache" / "apps" / "trading" / "gmx_charts" / entry_file / "signal.npy"
    if result.returncode != 0 or not signal_path.exists():
        raise RuntimeError(
            f"{entry_file} failed to produce signal.npy\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return float(np.load(signal_path).reshape(-1)[0])


def get_signal(market: str, period: str, repo_root=None, entry_file: str = "indicators.tl") -> dict:
    """Returns {"action": "BUY" | "SELL" | "HOLD", "raw": float | None,
    "error": str | None}. Never raises — this is the safe entry point
    for a live trading loop to call every tick.
    """
    try:
        raw = run_inference(market, period, repo_root, entry_file=entry_file)
    except RuntimeError as e:
        print(f"[signal_agent] WARNING: inference failed, defaulting to HOLD: {e}")
        return {"action": "HOLD", "raw": None, "error": str(e)}

    if raw > 0.0:
        action = "BUY"
    elif raw < 0.0:
        action = "SELL"
    else:
        action = "HOLD"
    return {"action": action, "raw": raw, "error": None}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--period", required=True)
    args = parser.parse_args()

    result = get_signal(args.market, args.period)
    print(result)

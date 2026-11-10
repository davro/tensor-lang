#!/usr/bin/env python3
"""Does apps/trading/gmx_charts/indicators.tl produce the right signal when `window` is f64?

Recomputes the app's pipeline in NumPy (float64) from window.npy and compares it with the program's
saved signal.npy.

    # 1. (optional) no window.npy yet? write a deterministic random-walk one at the path the app loads:
    python3 tools/check_indicators_f64.py --make-window          # seed 4: correct signal is +1 (buy)
    python3 tools/check_indicators_f64.py --make-window 6        # seed 6: correct signal is -1 (sell)
    # 2. run the app from the repo root (it saves signal.npy relative to where you run it):
    python3 tensorlang.py apps/trading/gmx_charts/indicators.tl
    # 3. compare:
    python3 tools/check_indicators_f64.py [signal.npy] [window.npy]

Note: build.sh clears cache/, so create and use window.npy in the same sitting.
Exit status 0 = signal matches NumPy, 1 = it does not.
"""
import sys
from pathlib import Path
import numpy as np

WINDOW = Path("cache/apps/trading/gmx_charts/indicators.tl/window.npy")


def reference(window):
    w = window.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sma_fast = w[190:200, 4:5].mean(axis=0)
        sma_slow = w[170:200, 4:5].mean(axis=0)
        prev, curr = w[185:199, 4:5], w[186:200, 4:5]
        gains = np.maximum(curr - prev, 0.0)
        losses = np.maximum(prev - curr, 0.0)
        rs = gains.mean(axis=0) / losses.mean(axis=0)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        up, down = (sma_fast > sma_slow), (sma_fast < sma_slow)
        not_ob, not_os = (rsi < 70.0), (rsi > 30.0)
        composite = (up & not_ob).astype(np.float64) - (down & not_os).astype(np.float64)
    return {"sma_fast": sma_fast, "sma_slow": sma_slow, "rsi": rsi,
            "is_uptrend": up, "is_downtrend": down, "composite": composite}


def main(argv):
    if argv[:1] == ["--make-window"]:
        seed = int(argv[1]) if len(argv) > 1 else 4
        rng = np.random.default_rng(seed)
        steps = rng.normal(0.0, 1.0, size=(200, 8)).cumsum(axis=0)
        w = (100.0 + steps).astype(np.float64)
        WINDOW.parent.mkdir(parents=True, exist_ok=True)
        np.save(WINDOW, w)
        print(f"wrote {WINDOW} {w.shape} {w.dtype} (seed {seed}); correct composite = {reference(w)['composite'].tolist()}")
        return 0
    if argv:
        signal_path = Path(argv[0])
    else:
        # the app saves signal.npy under cache/<path of the .tl file>/ (or the cwd); take the newest one
        found = sorted(list(Path(".").glob("signal.npy")) + list(Path("cache").glob("**/signal.npy")),
                       key=lambda q: q.stat().st_mtime) if Path("cache").exists() or Path("signal.npy").exists() else []
        if not found:
            print("no signal.npy found: run the app first, or pass its path"); return 1
        signal_path = found[-1]
        print(f"using {signal_path}")
    window_path = Path(argv[1]) if len(argv) > 1 else WINDOW
    window = np.load(window_path)
    ref = reference(window)
    got = np.load(signal_path)
    print(f"window {window.shape} {window.dtype}; signal {got.shape} {got.dtype}: {got.tolist()}")
    for k, v in ref.items():
        print(f"  numpy {k:13s} = {np.asarray(v).ravel().tolist()}")
    want = ref["composite"]
    ok = got.shape == want.shape and np.allclose(got, want, equal_nan=True)
    print("MATCHES NumPy" if ok else f"DOES NOT MATCH NumPy (expected {want.tolist()})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

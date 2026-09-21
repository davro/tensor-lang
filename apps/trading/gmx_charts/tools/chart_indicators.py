"""
apps/trading/gmx_charts/tools/chart_indicators.py

Registry of common CHART overlay indicators — plain pandas, no
TensorLang, no connection to indicators.tl's trading signal at all.
These exist so the chart looks like something you'd recognize from any
trading platform (SMA/EMA/Bollinger lines on the price chart); the
actual BUY/SELL/HOLD decision still lives entirely in indicators.tl.

Keeping these separate matters: a chart overlay just needs to look
sensible to a human eye, but a value indicators.tl uses to trigger a
trade needs to be provably correct against the same fixed-shape tensor
math every time. Mixing "what the chart shows" and "what the algo
decides" into one code path would blur that distinction — this file
is named chart_indicators.py rather than indicators.py specifically to
keep that boundary obvious at a glance.

Adding a new overlay indicator: add ONE entry to OVERLAY_INDICATORS.
tools/chart_server.py generates its settings checklist and its overlay
rendering entirely from this dict — nothing else needs to change.
"""
import numpy as np
import pandas as pd


def _sma(closes: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(closes).rolling(window).mean().values


def _ema(closes: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(closes).ewm(span=window, adjust=False).mean().values


def _bollinger(closes: np.ndarray, window: int, n_std: float):
    s = pd.Series(closes)
    mid = s.rolling(window).mean()
    std = s.rolling(window).std()
    return mid.values, (mid + n_std * std).values, (mid - n_std * std).values


# Each entry:
#   label     - shown in the chart-settings checklist
#   min_bars  - don't attempt this indicator on fewer bars than it needs
#               (e.g. no SMA 50 on a freshly-backfilled 30-candle window)
#   compute   - closes (1D array) -> {trace_name: values array}, values
#               may contain NaN for the warm-up period; Plotly skips
#               NaN points automatically, so no special handling needed
#   colors    - trace_name -> hex color
#   dash      - optional, trace_name -> Plotly dash style (e.g. "dot"),
#               only needed for traces that shouldn't be a solid line
OVERLAY_INDICATORS = {
    "sma20": {
        "label": "SMA 20",
        "min_bars": 20,
        "compute": lambda closes: {"SMA 20": _sma(closes, 20)},
        "colors": {"SMA 20": "#1f77b4"},
        "dash": {},
    },
    "sma50": {
        "label": "SMA 50",
        "min_bars": 50,
        "compute": lambda closes: {"SMA 50": _sma(closes, 50)},
        "colors": {"SMA 50": "#ff7f0e"},
        "dash": {},
    },
    "ema20": {
        "label": "EMA 20",
        "min_bars": 20,
        "compute": lambda closes: {"EMA 20": _ema(closes, 20)},
        "colors": {"EMA 20": "#2ca02c"},
        "dash": {},
    },
    "ema50": {
        "label": "EMA 50",
        "min_bars": 50,
        "compute": lambda closes: {"EMA 50": _ema(closes, 50)},
        "colors": {"EMA 50": "#d62728"},
        "dash": {},
    },
    "bb20": {
        "label": "Bollinger Bands (20, 2)",
        "min_bars": 20,
        "compute": lambda closes: dict(zip(
            ("BB Mid", "BB Upper", "BB Lower"), _bollinger(closes, 20, 2.0)
        )),
        "colors": {"BB Mid": "#9467bd", "BB Upper": "#9467bd", "BB Lower": "#9467bd"},
        "dash": {"BB Upper": "dot", "BB Lower": "dot"},
    },
}

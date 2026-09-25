"""
RSI (Relative Strength Index) implementation.

Used two ways in this bot:
1. On the signal timeframes (5m/15m/60m) as part of entry/exit logic.
2. On a higher timeframe (e.g. daily) purely as a risk filter — trades
   are throttled or vetoed when higher-timeframe RSI is at an extreme,
   regardless of what the lower-timeframe signal says.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RSIResult:
    values: List[float]  # RSI series, same length as input closes (NaN for warmup)

    @property
    def latest(self) -> Optional[float]:
        for v in reversed(self.values):
            if v == v:  # not NaN
                return v
        return None


def compute_rsi(closes: List[float], period: int = 14) -> RSIResult:
    """
    Standard Wilder's RSI.

    closes: list of close prices, oldest first.
    """
    n = len(closes)
    if n < period + 1:
        return RSIResult(values=[float("nan")] * n)

    closes_arr = np.asarray(closes, dtype=float)
    deltas = np.diff(closes_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.empty(n)
    avg_loss = np.empty(n)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    # seed with simple average over the first `period` deltas
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rsi = np.empty(n)
    rsi[:] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_vals = 100 - (100 / (1 + rs))
    # where avg_loss is 0 and avg_gain > 0 -> RSI = 100
    rsi_vals = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi_vals)
    # where both are 0 -> RSI = 50 (flat market)
    rsi_vals = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi_vals)
    rsi[period:] = rsi_vals[period:]

    return RSIResult(values=rsi.tolist())

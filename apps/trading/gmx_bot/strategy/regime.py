"""
Regime layer: reads higher-timeframe RSI (daily/weekly/monthly by
default) to classify a symbol's medium/long-term bias as BULLISH,
BEARISH, or NEUTRAL, then uses that bias to adjust an already-computed
MultiTimeframeSignal — scaling down or vetoing entries that fight the
larger trend.

This is a FILTER, not a signal generator: it never turns a FLAT signal
into a LONG/SHORT one, and it never flips a signal's direction. It only
narrows what generate_multi_timeframe_signal already decided, using the
exact same Direction/MultiTimeframeSignal shapes that function already
returns — so it drops into main.py as one extra step per symbol, with
no changes needed in risk/ or execution/ (RiskManager.evaluate() only
ever reads symbol/direction/risk_scale/reason, all of which are
preserved here).

Why a separate module instead of just adding 1d/1w/1M to
trend_timeframes? generate_multi_timeframe_signal's veto/scale logic
in signal.py is written for MEAN-REVERSION RSI — it looks for extremes
(oversold/overbought) and assumes price reverts from them. That's the
right model for the entry timeframes, but not for weekly/monthly RSI on
a genuinely trending asset: RSI there can sit at 60-65 for weeks
without ever giving a classic "overbought" reading, so treating it the
same way just means fighting a real trend under the label "risk
management." Regime classification here instead asks a simpler
question — "which side of neutral (50) is price structurally living
on" — via a vote across regime_timeframes against a wide neutral band,
not an extreme.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from config.settings import RSIConfig
from data.market_data import MarketDataProvider
from indicators.rsi import compute_rsi
from strategy.signal import Direction, MultiTimeframeSignal

# Defaults live here (not only as function-arg defaults) so they're easy
# to find and tune without hunting through call sites. Move these into
# config/settings.py (e.g. settings.market.regime_timeframes,
# settings.rsi.regime_bullish / regime_bearish) once you're happy with
# them — main.py's call already uses getattr(...) fallbacks to these
# same values, so adding them to settings.py is a drop-in, not a
# required step.
DEFAULT_REGIME_TIMEFRAMES = ["1d", "1w", "1M"]
DEFAULT_BULLISH_RSI = 55.0
DEFAULT_BEARISH_RSI = 45.0
DEFAULT_REGIME_MIN_AGREEMENT = 2
DEFAULT_COUNTER_TREND_SCALE = 0.3  # how much to shrink a counter-trend entry, not veto it outright

# 1w/1M are resampled from daily candles in market_data.py (GMX's API has
# no native weekly/monthly period), so their fetch cost scales directly
# with how much daily history gets requested. The blanket
# max(100, rsi_cfg.period * 3) floor used elsewhere in this codebase
# suits entry/trend timeframes (cheap, native GMX periods) but would mean
# requesting 100 monthly bars = ~8 years of daily candles, refetched every
# cycle, for every symbol. These overrides keep it to a still-generous
# window of real history without that cost.
CANDLE_LIMIT_OVERRIDES = {
    "1w": 60,   # ~14 months of weekly bars
    "1M": 36,   # 3 years of monthly bars
}


class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class RegimeReading:
    regime: Regime
    rsis: Dict[str, Optional[float]]  # {timeframe: rsi or None}
    reason: str
    votes: int = 0   # how many regime_timeframes agreed with the winning direction
    total: int = 0   # how many regime_timeframes were evaluated

    def display(self) -> str:
        rsi_str = " ".join(
            f"{tf}={v:.1f}" if v is not None else f"{tf}=n/a" for tf, v in self.rsis.items()
        )
        return f"regime[{rsi_str}] -> {self.regime.value} ({self.reason})"


def _closes(candles):
    return [c.close for c in candles]


def classify_regime(
    symbol: str,
    data: MarketDataProvider,
    rsi_cfg: RSIConfig,
    regime_timeframes: List[str] = None,
    bullish_rsi: float = DEFAULT_BULLISH_RSI,
    bearish_rsi: float = DEFAULT_BEARISH_RSI,
    min_agreement: int = DEFAULT_REGIME_MIN_AGREEMENT,
) -> RegimeReading:
    """
    Votes across regime_timeframes (default 1d/1w/1M — deliberately
    separate from signal.py's trend_timeframes, which stays focused on
    the 4h/1d veto/scale job it already does for entry timeframes).

    bullish_rsi/bearish_rsi default to 55/45: a deliberately wide
    neutral band around 50 so one borderline reading can't flip the
    whole regime on its own — min_agreement then requires at least 2 of
    the (default 3) regime timeframes to agree before committing to a
    bias at all.
    """
    regime_timeframes = regime_timeframes or DEFAULT_REGIME_TIMEFRAMES
    rsis = {}
    for tf in regime_timeframes:
        limit = CANDLE_LIMIT_OVERRIDES.get(tf, max(100, rsi_cfg.period * 3))
        candles = data.get_candles(symbol, tf, limit=limit)
        rsi = compute_rsi(_closes(candles), rsi_cfg.period).latest if len(candles) >= rsi_cfg.period + 2 else None
        rsis[tf] = rsi

    bullish_votes = sum(1 for v in rsis.values() if v is not None and v >= bullish_rsi)
    bearish_votes = sum(1 for v in rsis.values() if v is not None and v <= bearish_rsi)

    if bullish_votes >= min_agreement and bullish_votes > bearish_votes:
        return RegimeReading(
            Regime.BULLISH, rsis,
            f"{bullish_votes}/{len(regime_timeframes)} regime timeframes >= {bullish_rsi}",
            votes=bullish_votes, total=len(regime_timeframes),
        )
    if bearish_votes >= min_agreement and bearish_votes > bullish_votes:
        return RegimeReading(
            Regime.BEARISH, rsis,
            f"{bearish_votes}/{len(regime_timeframes)} regime timeframes <= {bearish_rsi}",
            votes=bearish_votes, total=len(regime_timeframes),
        )
    return RegimeReading(
        Regime.NEUTRAL, rsis,
        f"no regime consensus (bull={bullish_votes} bear={bearish_votes}, need {min_agreement})",
        votes=max(bullish_votes, bearish_votes), total=len(regime_timeframes),
    )


def apply_regime_filter(
    signal: MultiTimeframeSignal,
    regime: RegimeReading,
    counter_trend_scale: float = DEFAULT_COUNTER_TREND_SCALE,
    veto_counter_trend: bool = False,
) -> MultiTimeframeSignal:
    """
    Adjusts an already-generated signal based on regime alignment:

      - FLAT in  -> FLAT out, always. This layer never creates a trade.
      - Aligned with regime (e.g. LONG signal + BULLISH regime)
        -> unchanged direction/scale, reason annotated for visibility.
      - Fights the regime (e.g. LONG signal + BEARISH regime)
        -> risk_scale shrunk to counter_trend_scale (default 0.3x), or
        vetoed to FLAT entirely if veto_counter_trend=True.
      - Regime is NEUTRAL (no consensus) -> signal passed through as-is;
        there's no established bias to check alignment against.

    Returns the same MultiTimeframeSignal shape (symbol, direction,
    risk_scale, reason, entry_rsis, trend_rsis) so it drops straight
    into the existing pipeline wherever generate_multi_timeframe_signal
    used to be returned directly.
    """
    if signal.direction == Direction.FLAT:
        return signal

    aligned = (
        (signal.direction == Direction.LONG and regime.regime == Regime.BULLISH)
        or (signal.direction == Direction.SHORT and regime.regime == Regime.BEARISH)
    )
    counter = (
        (signal.direction == Direction.LONG and regime.regime == Regime.BEARISH)
        or (signal.direction == Direction.SHORT and regime.regime == Regime.BULLISH)
    )

    if aligned:
        return MultiTimeframeSignal(
            signal.symbol, signal.direction, signal.risk_scale,
            f"{signal.reason} | regime-aligned ({regime.reason})",
            signal.entry_rsis, signal.trend_rsis,
        )

    if counter:
        if veto_counter_trend:
            return MultiTimeframeSignal(
                signal.symbol, Direction.FLAT, 0.0,
                f"vetoed: counter-trend to {regime.regime.value} regime ({regime.reason})",
                signal.entry_rsis, signal.trend_rsis,
            )
        scaled = min(signal.risk_scale, counter_trend_scale)
        return MultiTimeframeSignal(
            signal.symbol, signal.direction, scaled,
            f"{signal.reason} | counter-trend to {regime.regime.value} regime, scaled to {scaled}x ({regime.reason})",
            signal.entry_rsis, signal.trend_rsis,
        )

    # NEUTRAL regime: no established bias to check alignment against —
    # pass the entry-timeframe signal through unchanged.
    return signal
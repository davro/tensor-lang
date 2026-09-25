"""
Signal generation.

Entries/exits are driven by RSI on the lower timeframes (5m/15m/60m).
A higher-timeframe RSI (e.g. daily) acts purely as a risk filter: it can
veto a trade or scale down its size, but it never generates a signal on
its own.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from config.settings import RSIConfig
from data.market_data import Candle, MarketDataProvider
from indicators.rsi import compute_rsi


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    symbol: str
    timeframe: str
    direction: Direction
    entry_rsi: Optional[float]
    risk_rsi: Optional[float]
    risk_scale: float  # 0.0 (veto) to 1.0 (full size), applied by risk manager
    reason: str


def _closes(candles: List[Candle]) -> List[float]:
    return [c.close for c in candles]


def generate_signal(
    symbol: str,
    timeframe: str,
    data: MarketDataProvider,
    rsi_cfg: RSIConfig,
    risk_timeframe: str,
) -> Signal:
    candles = data.get_candles(symbol, timeframe, limit=max(200, rsi_cfg.period * 3))
    if len(candles) < rsi_cfg.period + 2:
        return Signal(symbol, timeframe, Direction.FLAT, None, None, 0.0, "insufficient data")

    entry_rsi_series = compute_rsi(_closes(candles), rsi_cfg.period)
    entry_rsi = entry_rsi_series.latest

    risk_candles = data.get_candles(symbol, risk_timeframe, limit=max(100, rsi_cfg.period * 3))
    risk_rsi_series = compute_rsi(_closes(risk_candles), rsi_cfg.period) if risk_candles else None
    risk_rsi = risk_rsi_series.latest if risk_rsi_series else None

    if entry_rsi is None:
        return Signal(symbol, timeframe, Direction.FLAT, None, risk_rsi, 0.0, "rsi warming up")

    # --- Entry logic (lower timeframe) ---
    if entry_rsi <= rsi_cfg.oversold:
        direction = Direction.LONG
        reason = f"{timeframe} RSI {entry_rsi:.1f} <= oversold {rsi_cfg.oversold}"
    elif entry_rsi >= rsi_cfg.overbought:
        direction = Direction.SHORT
        reason = f"{timeframe} RSI {entry_rsi:.1f} >= overbought {rsi_cfg.overbought}"
    else:
        return Signal(symbol, timeframe, Direction.FLAT, entry_rsi, risk_rsi, 0.0, "no edge")

    # --- Risk filter (higher timeframe) ---
    risk_scale = 1.0
    if risk_rsi is not None:
        if direction == Direction.LONG and risk_rsi >= rsi_cfg.risk_overbought:
            return Signal(
                symbol, timeframe, Direction.FLAT, entry_rsi, risk_rsi, 0.0,
                f"vetoed: {risk_timeframe} RSI {risk_rsi:.1f} overbought, won't add longs",
            )
        if direction == Direction.SHORT and risk_rsi <= rsi_cfg.risk_oversold:
            return Signal(
                symbol, timeframe, Direction.FLAT, entry_rsi, risk_rsi, 0.0,
                f"vetoed: {risk_timeframe} RSI {risk_rsi:.1f} oversold, won't add shorts",
            )
        # Scale down size as higher-TF RSI approaches the opposite extreme
        # relative to the trade direction (reduces size when going against
        # a strong higher-timeframe trend).
        if direction == Direction.LONG and risk_rsi < rsi_cfg.risk_oversold + 15:
            risk_scale = 0.5
        if direction == Direction.SHORT and risk_rsi > rsi_cfg.risk_overbought - 15:
            risk_scale = 0.5

    return Signal(symbol, timeframe, direction, entry_rsi, risk_rsi, risk_scale, reason)


@dataclass
class MultiTimeframeSignal:
    """
    One combined signal per symbol, built from RSI across several entry
    timeframes (voted, to avoid a single noisy timeframe firing alone)
    plus multiple higher timeframes as a trend filter (any of them can
    veto or scale down the trade — the strictest one wins).

    Has the same 4 attributes RiskManager.evaluate() actually reads
    (symbol, direction, risk_scale, reason), so it drops into the
    existing risk/execution pipeline without changes there.
    """
    symbol: str
    direction: Direction
    risk_scale: float
    reason: str
    entry_rsis: dict            # {timeframe: rsi or None}
    trend_rsis: dict            # {timeframe: rsi or None}

    def display(self) -> str:
        entry_str = " ".join(
            f"{tf}={v:.1f}" if v is not None else f"{tf}=n/a" for tf, v in self.entry_rsis.items()
        )
        trend_str = " ".join(
            f"{tf}={v:.1f}" if v is not None else f"{tf}=n/a" for tf, v in self.trend_rsis.items()
        )
        return f"{self.symbol} | entry[{entry_str}] trend[{trend_str}] -> {self.direction.value} ({self.reason})"


def generate_multi_timeframe_signal(
    symbol: str,
    data: MarketDataProvider,
    rsi_cfg: RSIConfig,
    entry_timeframes: List[str],
    trend_timeframes: List[str],
    min_agreement: int,
) -> MultiTimeframeSignal:
    entry_rsis = {}
    for tf in entry_timeframes:
        candles = data.get_candles(symbol, tf, limit=max(200, rsi_cfg.period * 3))
        rsi = compute_rsi(_closes(candles), rsi_cfg.period).latest if len(candles) >= rsi_cfg.period + 2 else None
        entry_rsis[tf] = rsi

    trend_rsis = {}
    for tf in trend_timeframes:
        candles = data.get_candles(symbol, tf, limit=max(100, rsi_cfg.period * 3))
        rsi = compute_rsi(_closes(candles), rsi_cfg.period).latest if len(candles) >= rsi_cfg.period + 2 else None
        trend_rsis[tf] = rsi

    long_votes = sum(1 for v in entry_rsis.values() if v is not None and v <= rsi_cfg.oversold)
    short_votes = sum(1 for v in entry_rsis.values() if v is not None and v >= rsi_cfg.overbought)

    if long_votes >= min_agreement and long_votes > short_votes:
        direction = Direction.LONG
        reason = f"{long_votes}/{len(entry_timeframes)} entry timeframes oversold"
    elif short_votes >= min_agreement and short_votes > long_votes:
        direction = Direction.SHORT
        reason = f"{short_votes}/{len(entry_timeframes)} entry timeframes overbought"
    else:
        return MultiTimeframeSignal(
            symbol, Direction.FLAT, 0.0,
            f"no consensus (long_votes={long_votes} short_votes={short_votes}, need {min_agreement})",
            entry_rsis, trend_rsis,
        )

    # --- Trend filter across ALL higher timeframes: strictest one wins ---
    risk_scale = 1.0
    for tf, trend_rsi in trend_rsis.items():
        if trend_rsi is None:
            continue
        if direction == Direction.LONG and trend_rsi >= rsi_cfg.risk_overbought:
            return MultiTimeframeSignal(
                symbol, Direction.FLAT, 0.0,
                f"vetoed: {tf} RSI {trend_rsi:.1f} overbought, won't add longs",
                entry_rsis, trend_rsis,
            )
        if direction == Direction.SHORT and trend_rsi <= rsi_cfg.risk_oversold:
            return MultiTimeframeSignal(
                symbol, Direction.FLAT, 0.0,
                f"vetoed: {tf} RSI {trend_rsi:.1f} oversold, won't add shorts",
                entry_rsis, trend_rsis,
            )
        if direction == Direction.LONG and trend_rsi < rsi_cfg.risk_oversold + 15:
            risk_scale = min(risk_scale, 0.5)
        if direction == Direction.SHORT and trend_rsi > rsi_cfg.risk_overbought - 15:
            risk_scale = min(risk_scale, 0.5)

    return MultiTimeframeSignal(symbol, direction, risk_scale, reason, entry_rsis, trend_rsis)


@dataclass
class ExitSignal:
    should_exit: bool
    rsi: Optional[float]
    reason: str


def generate_exit_signal(
    symbol: str,
    timeframe: str,
    position_direction: Direction,
    data: MarketDataProvider,
    rsi_cfg: RSIConfig,
    regime_bias: Optional[str] = None,
) -> ExitSignal:
    """
    Primary exit trigger: RSI reversion. A long entered on oversold RSI is
    closed once RSI climbs back to/above the neutral line; a short entered
    on overbought RSI is closed once RSI falls back to/below neutral.

    This does not replace a hard stop-loss (see execution/gmx_client.py's
    stop_loss_pct handling) — it's the "normal" way out of a trade that's
    working, or one that's stalled without yet hitting a hard loss limit.

    regime_bias (optional: "bullish" / "bearish" / "neutral" / None, as
    produced by strategy/regime.py — passed as a plain string rather than
    importing the Regime enum, to avoid a signal.py <-> regime.py import
    cycle) shifts how far RSI has to revert before exiting:

      - Aligned with the regime (e.g. a LONG held while regime_bias is
        "bullish") gets MORE room — exit_neutral is pushed further away
        in the trade's favor, since a real uptrend can keep entry-
        timeframe RSI elevated for a while without the move being over.
      - Fighting the regime (e.g. a LONG held while regime_bias is
        "bearish") gets LESS room — exit_neutral is pulled closer, since
        there's no larger trend backing the trade up if it stalls.
      - None or "neutral" leaves exit_neutral exactly as rsi_cfg defines
        it — this parameter is purely additive, omitting it reproduces
        the exact behavior this function had before it existed.
    """
    candles = data.get_candles(symbol, timeframe, limit=max(200, rsi_cfg.period * 3))
    if len(candles) < rsi_cfg.period + 2:
        return ExitSignal(False, None, "insufficient data")

    rsi_series = compute_rsi(_closes(candles), rsi_cfg.period)
    rsi = rsi_series.latest
    if rsi is None:
        return ExitSignal(False, None, "rsi warming up")

    exit_neutral = rsi_cfg.exit_neutral
    regime_note = ""
    aligned = (
        (position_direction == Direction.LONG and regime_bias == "bullish")
        or (position_direction == Direction.SHORT and regime_bias == "bearish")
    )
    counter = (
        (position_direction == Direction.LONG and regime_bias == "bearish")
        or (position_direction == Direction.SHORT and regime_bias == "bullish")
    )
    if aligned:
        shift = 10  # more room to run
        exit_neutral = exit_neutral + shift if position_direction == Direction.LONG else exit_neutral - shift
        regime_note = f", widened for {regime_bias} regime alignment"
    elif counter:
        shift = 5  # less room, exit sooner
        exit_neutral = exit_neutral - shift if position_direction == Direction.LONG else exit_neutral + shift
        regime_note = f", tightened \u2014 counter-trend to {regime_bias} regime"

    if position_direction == Direction.LONG and rsi >= exit_neutral:
        return ExitSignal(True, rsi, f"{timeframe} RSI reverted to {rsi:.1f} (>= {exit_neutral}), closing long{regime_note}")
    if position_direction == Direction.SHORT and rsi <= exit_neutral:
        return ExitSignal(True, rsi, f"{timeframe} RSI reverted to {rsi:.1f} (<= {exit_neutral}), closing short{regime_note}")

    return ExitSignal(False, rsi, f"{timeframe} RSI {rsi:.1f}, still in trend, holding{regime_note}")
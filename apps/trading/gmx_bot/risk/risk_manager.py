"""
Risk management.

Every proposed trade passes through here before it's allowed to become
an order. This is the one place hard limits live — the strategy layer
should never be trusted to enforce them itself.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from config.settings import RiskConfig
from strategy.signal import Direction, Signal


@dataclass
class OpenPosition:
    symbol: str
    direction: Direction
    notional_usd: float
    unrealized_pnl_usd: float = 0.0
    market_address: Optional[str] = None          # GMX market contract address, for closing
    collateral_amount_raw: Optional[int] = None    # smallest-unit collateral (e.g. USDC, 6 decimals)


@dataclass
class RiskDecision:
    approved: bool
    size_usd: float
    leverage: float
    stop_loss_pct: float
    take_profit_pct: float
    reason: str


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.realized_pnl_today_usd: float = 0.0
        self.trading_halted: bool = False

    def record_realized_pnl(self, pnl_usd: float) -> None:
        self.realized_pnl_today_usd += pnl_usd
        if self.realized_pnl_today_usd <= -abs(self.cfg.max_daily_loss_usd):
            self.trading_halted = True

    def reset_daily(self) -> None:
        self.realized_pnl_today_usd = 0.0
        self.trading_halted = False

    def evaluate(
        self,
        signal: Signal,
        open_positions: List[OpenPosition],
        account_equity_usd: float,
    ) -> RiskDecision:
        deny = lambda reason: RiskDecision(False, 0.0, 0.0, 0.0, 0.0, reason)

        if self.trading_halted:
            return deny(f"daily loss limit hit (${self.realized_pnl_today_usd:.2f}); trading halted")

        if signal.direction == Direction.FLAT:
            return deny("no signal")

        if signal.risk_scale <= 0.0:
            return deny(signal.reason)

        if account_equity_usd <= 0:
            return deny(
                "account equity is $0.00 — wallet has no USDC collateral yet "
                "(or the balance/position read failed); nothing to size a trade against"
            )

        if len(open_positions) >= self.cfg.max_open_positions:
            return deny(f"max open positions ({self.cfg.max_open_positions}) reached")

        existing = next((p for p in open_positions if p.symbol == signal.symbol), None)
        if existing and existing.direction != signal.direction:
            return deny(
                f"existing {existing.direction.value} position on {signal.symbol}; "
                f"close it before flipping direction"
            )

        total_exposure = sum(p.notional_usd for p in open_positions)
        if total_exposure >= self.cfg.max_account_exposure_usd:
            return deny(
                f"account exposure ${total_exposure:.2f} already at/above cap "
                f"${self.cfg.max_account_exposure_usd:.2f}"
            )

        remaining_capacity = self.cfg.max_account_exposure_usd - total_exposure
        size_usd = min(self.cfg.max_position_usd, remaining_capacity) * signal.risk_scale
        if size_usd <= 0:
            return deny("no remaining risk capacity")

        # Required leverage is whatever makes collateral == size/leverage
        # fit within available equity, capped by the configured max.
        # If even max_leverage can't fit this size within equity, deny
        # rather than silently over-leveraging.
        min_leverage_needed = size_usd / max(account_equity_usd, 1e-9)
        if min_leverage_needed > self.cfg.max_leverage:
            return deny(
                f"position size ${size_usd:.2f} would need "
                f"{min_leverage_needed:.2f}x leverage, above cap "
                f"{self.cfg.max_leverage}x for equity ${account_equity_usd:.2f}"
            )
        leverage = max(min_leverage_needed, 1.0)
        leverage = min(leverage, self.cfg.max_leverage)

        return RiskDecision(
            approved=True,
            size_usd=round(size_usd, 2),
            leverage=round(leverage, 2),
            stop_loss_pct=self.cfg.default_stop_loss_pct,
            take_profit_pct=self.cfg.default_take_profit_pct,
            reason=signal.reason,
        )

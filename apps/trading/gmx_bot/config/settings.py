"""
Central configuration for the TensorLang trading bot.

All secrets (private key, RPC URL) come from environment variables —
never hardcode them here or commit them to source control.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChainConfig:
    chain: str = "arbitrum"
    chain_id: int = 42161  # Arbitrum One
    rpc_url: str = field(default_factory=lambda: os.environ.get("ARBITRUM_RPC_URL", ""))
    # NEVER put a real private key in code/config files. Load from env only.
    private_key: str = field(default_factory=lambda: os.environ.get("GMX_PRIVATE_KEY", ""))
    wallet_address: str = field(default_factory=lambda: os.environ.get("GMX_WALLET_ADDRESS", ""))


@dataclass
class MarketConfig:
    # GMX V2 market symbols you want the bot to watch/trade.
    # Expanded using the actual market list pulled from
    # scripts/list_markets.py against this account's chain (125 markets
    # found on Arbitrum). Includes GMX's TradFi additions (commodities,
    # SPCX, QQQ, SPY) confirmed live in 2026.
    #
    # DELIBERATELY EXCLUDED: AAVE, BNB, LINK, PEPE, SOL, UNI, WIF — each
    # of these has TWO separate GMX markets sharing the same ticker on
    # this chain. MarketResolver refuses to guess between them; add any
    # of these back once you've set an explicit address in
    # market_address_overrides below.
    symbols: List[str] = field(default_factory=lambda: [
        # Crypto majors (unambiguous — single market each)
        "BTC/USD", "ETH/USD", "ARB/USD", "DOGE/USD", "LTC/USD", "AVAX/USD",
        "ATOM/USD", "XRP/USD", "ADA/USD", "DOT/USD", "NEAR/USD", "SUI/USD",
        "OP/USD", "TRX/USD", "TIA/USD", "FIL/USD", "ICP/USD", "XLM/USD",
        "DYDX/USD", "INJ/USD", "APT/USD", "RENDER/USD", "JUP/USD",
        "ONDO/USD", "TAO/USD", "PENDLE/USD", "SEI/USD", "HBAR/USD",
        "ENA/USD", "EIGEN/USD", "MORPHO/USD", "VIRTUAL/USD", "STX/USD",
        # TradFi (confirmed live on GMX; different risk profile — see notes below)
        "GOLD/USD", "SILVER/USD", "XAUT/USD", "NATGAS/USD",
        "WTIOIL/USD", "BRENTOIL/USD", "SPCX/USD", "QQQ/USD", "SPY/USD",
    ])
    # GMX also added a TradFi category in 2026: commodities (gold, silver,
    # oil, natgas) and even synthetic stocks/indices (SPCX, QQQ, SPY),
    # traded through the SAME order flow as crypto — no separate
    # execution path needed. Before adding any of these to `symbols`
    # above, run `python3 scripts/list_markets.py` to get GMX's exact
    # market_symbol string (docs mention both a name like "GOLD" and a
    # ticker like "XAU" as search terms — only one of those is the
    # actual field MarketResolver matches against).
    #
    # Two things worth knowing before trading these with an RSI bot:
    # - Leverage/fees shift between "on-hours" and "off-hours" sessions
    #   for most TradFi markets (e.g. gold: 100x on-hours -> 25x off-hours).
    #   This bot's own max_leverage cap (see RiskConfig) is already well
    #   under any of those, so no risk-manager change is needed — just
    #   know GMX's own limits move under you throughout the day.
    # - SPCX/USD is a pre-IPO synthetic market with limited open-interest
    #   caps — much thinner liquidity than BTC/ETH.
    #
    # Truly non-GMX assets (individual stocks beyond SPCX, FX pairs) still
    # have no execution path — that's what traditional_watch_symbols below
    # is for: signal-only tracking via a separate data feed, once wired up.
    traditional_watch_symbols: List[str] = field(default_factory=list)
    # For symbols where GMX has more than one market sharing the same
    # ticker (seen on Arbitrum: AAVE, BNB, LINK, PEPE, SOL, UNI, WIF —
    # run scripts/list_markets.py to check current duplicates), the bot
    # refuses to guess which one to trade. Set the exact address here
    # after checking each candidate market yourself, e.g.:
    #   market_address_overrides = {"SOL/USD": "0xcf083d35AD306A042d4Fb312fCdd8228b52b82f8"}
    market_address_overrides: Dict[str, str] = field(default_factory=dict)
    # Lower timeframes used for entries, voted on together (see
    # min_entry_agreement below) rather than each firing its own
    # independent signal — avoids the same symbol getting 2-3 separate,
    # sometimes conflicting, order attempts in one cycle.
    signal_timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])
    # How many of the timeframes above must agree (all oversold, or all
    # overbought) before a direction is even considered. 2 of 3 filters
    # out a single noisy timeframe spiking alone.
    min_entry_agreement: int = 2
    # Higher timeframes used purely as a trend filter — any one of them
    # can veto or scale down a trade, but none of them generate a signal
    # on their own. 4h catches medium-term trend, 1d catches the
    # dominant daily trend; requiring both to not disagree with the
    # trade is stricter than just checking 1d alone.
    trend_timeframes: List[str] = field(default_factory=lambda: ["4h", "1d"])
    # Timeframe checked for RSI-reversion exits. A position doesn't
    # currently remember which timeframes triggered its entry, so exits
    # are evaluated on a single designated timeframe for consistency.
    exit_timeframe: str = "15m"


@dataclass
class RSIConfig:
    period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    # Higher-timeframe RSI thresholds used to throttle/veto trades
    risk_overbought: float = 75.0
    risk_oversold: float = 25.0
    # Primary exit trigger: close once RSI reverts back across this line
    # (long closes when RSI climbs to/above it, short when RSI falls to/below it)
    exit_neutral: float = 50.0


@dataclass
class RiskConfig:
    # Hard caps — the bot should refuse to exceed these no matter what
    # the strategy signal says.
    max_leverage: float = 3.0
    max_position_usd: float = 200.0          # per-position cap in USD collateral*leverage
    max_account_exposure_usd: float = 500.0  # sum of all open notional
    max_daily_loss_usd: float = 50.0         # circuit breaker: stop trading for the day
    max_open_positions: int = 3
    default_stop_loss_pct: float = 2.0       # % move against position that triggers stop
    default_take_profit_pct: float = 4.0
    slippage_bps: int = 30                   # acceptable price slippage, in basis points


@dataclass
class ExecutionConfig:
    # SAFETY DEFAULT: bot builds and logs orders but does not submit them
    # on-chain until this is explicitly set to False. Flip only when you've
    # reviewed logged orders and are ready to trade with real funds.
    debug_mode: bool = True
    poll_interval_seconds: int = 30
    execution_fee_buffer_pct: float = 20.0   # buffer added to keeper execution fee estimate


@dataclass
class Settings:
    chain: ChainConfig = field(default_factory=ChainConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    rsi: RSIConfig = field(default_factory=RSIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def validate(self) -> None:
        if not self.chain.rpc_url:
            raise ValueError("ARBITRUM_RPC_URL is not set in the environment.")
        if not self.execution.debug_mode:
            if not self.chain.private_key:
                raise ValueError(
                    "GMX_PRIVATE_KEY is not set, but debug_mode is False. "
                    "Refusing to run live without signing credentials."
                )
            if not self.chain.wallet_address:
                raise ValueError("GMX_WALLET_ADDRESS is not set.")


settings = Settings()

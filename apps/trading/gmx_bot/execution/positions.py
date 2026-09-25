"""
Position and equity tracking, backed by eth_defi.gmx (read-only calls,
no signing needed — GMXConfig is built without a wallet for these reads).

Field names below were confirmed by reading the installed eth_defi
source directly (GetOpenPositions._process_rest_api_position), not
guessed from docs — those had drifted from the real API in an earlier
version of this file. If you upgrade web3-ethereum-defi later and this
starts logging "Unrecognized position shape" warnings, re-check that
method's source for renamed fields.

get_user_positions() returns a dict keyed like "BTC_long", with values
containing (among others): market_symbol, is_long, position_size (USD),
pnl_after_fees (USD), initial_collateral_amount (raw smallest-unit
USDC), and market (the GMX market contract address).

Equity here is approximated as (USDC wallet balance) + (sum of open
position sizes). That's conservative for position-sizing purposes, but
it is NOT your exact account value — check GMX's UI/API if you need that.
"""
import logging
import os
from typing import List, Optional

from config.settings import ChainConfig, settings
from execution.markets import USDC_ARBITRUM
from risk.risk_manager import OpenPosition
from strategy.signal import Direction

logger = logging.getLogger("execution.positions")

# Lets you exercise RiskManager.evaluate()'s sizing/scaling logic
# (including regime counter-trend scaling) without funding the wallet
# first — set DRY_RUN_EQUITY_USD in .env (or the environment) to a
# number, e.g. DRY_RUN_EQUITY_USD=1000. Only honored while
# settings.execution.debug_mode is True, so it can never mask a real
# equity read failure once live.
DRY_RUN_EQUITY_ENV = "DRY_RUN_EQUITY_USD"

ERC20_BALANCE_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


class PositionTracker:
    def __init__(self, chain_cfg: ChainConfig):
        self.chain_cfg = chain_cfg
        self._w3 = None
        self._gmx_config = None
        self._market_data = None

    def _ensure_connected(self) -> bool:
        if self._market_data is not None:
            return True
        if not self.chain_cfg.rpc_url or not self.chain_cfg.wallet_address:
            logger.warning("No RPC URL or wallet address configured; skipping position lookup.")
            return False

        from web3 import Web3
        from eth_defi.gmx.config import GMXConfig
        from eth_defi.gmx.data import GMXMarketData

        self._w3 = Web3(Web3.HTTPProvider(self.chain_cfg.rpc_url))
        # Read-only: no "wallet" (signer) needed just to look up positions,
        # and there's no "chain=" kwarg — chain is auto-detected from web3.
        self._gmx_config = GMXConfig(web3=self._w3, user_wallet_address=self.chain_cfg.wallet_address)
        self._market_data = GMXMarketData(self._gmx_config)
        return True

    def get_open_positions(self) -> List[OpenPosition]:
        if not self._ensure_connected():
            return []
        try:
            raw_positions = self._market_data.get_user_positions(self.chain_cfg.wallet_address)
        except Exception:
            logger.exception("Failed to fetch open positions from GMX")
            return []

        positions = []
        for key, raw in (raw_positions or {}).items():
            pos = _normalize_position(key, raw)
            if pos:
                positions.append(pos)
        return positions

    def get_account_equity_usd(self, open_positions: Optional[List[OpenPosition]] = None) -> float:
        """Pass in positions already fetched this cycle (from
        get_open_positions()) to avoid querying GMX for them twice."""
        if settings.execution.debug_mode:
            override = os.environ.get(DRY_RUN_EQUITY_ENV)
            if override:
                try:
                    return float(override)
                except ValueError:
                    logger.warning(
                        "%s=%r isn't a valid number, ignoring override",
                        DRY_RUN_EQUITY_ENV, override,
                    )

        if not self._ensure_connected():
            return 0.0

        usdc_balance = 0.0
        try:
            usdc = self._w3.eth.contract(
                address=self._w3.to_checksum_address(USDC_ARBITRUM), abi=ERC20_BALANCE_ABI
            )
            decimals = usdc.functions.decimals().call()
            raw_balance = usdc.functions.balanceOf(
                self._w3.to_checksum_address(self.chain_cfg.wallet_address)
            ).call()
            usdc_balance = raw_balance / (10 ** decimals)
        except Exception:
            logger.exception("Failed to fetch USDC wallet balance")

        if open_positions is None:
            open_positions = self.get_open_positions()
        position_notional = sum(p.notional_usd for p in open_positions)
        return usdc_balance + position_notional

    def debug_dump_raw_positions(self) -> None:
        """Run this manually to see the exact shape get_user_positions()
        returns on your installed eth_defi version, if you ever suspect
        the field names below have drifted after an upgrade."""
        if not self._ensure_connected():
            print("No wallet/RPC configured.")
            return
        print(self._market_data.get_user_positions(self.chain_cfg.wallet_address))


def _normalize_position(key: str, raw: dict) -> Optional[OpenPosition]:
    try:
        market_symbol = raw.get("market_symbol")
        is_long = raw.get("is_long")
        if market_symbol is None or is_long is None:
            logger.warning("Unrecognized position shape for %s, skipping: %s", key, raw)
            return None

        return OpenPosition(
            symbol=f"{market_symbol}/USD",
            direction=Direction.LONG if is_long else Direction.SHORT,
            notional_usd=float(raw.get("position_size", 0.0)),
            unrealized_pnl_usd=float(raw.get("pnl_after_fees", 0.0)),
            market_address=raw.get("market"),
            collateral_amount_raw=raw.get("initial_collateral_amount"),
        )
    except Exception:
        logger.exception("Failed to normalize position %s: %s", key, raw)
        return None
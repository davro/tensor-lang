"""
Execution layer: turns an approved RiskDecision into a real GMX V2 order.

Built on `web3-ethereum-defi` (eth_defi.gmx). The flow, confirmed against
the installed library's actual source (not just its docs, which lagged
the real API in an earlier version of this file):

  1. IncreaseOrder/DecreaseOrder.create_*_order(...) builds an UNSIGNED
     transaction (OrderResult.transaction) — it does not sign or send
     anything itself.
  2. HotWallet.sign_transaction_with_new_nonce(tx) signs it and assigns a nonce.
  3. get_tx_broadcast_data(signed) + web3.eth.send_raw_transaction(...)
     actually broadcasts it.

GMX orders are two-step and asynchronous even after that: broadcasting
creates the order on-chain, then a keeper executes it later against
oracle pricing. A returned tx_hash means "the order was submitted",
not "the trade filled" — track fills separately via positions.py.

debug_mode=True (the config default) builds the order and logs it
without signing or broadcasting anything — nothing touches the chain or
your funds until you flip it off deliberately.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import ChainConfig, ExecutionConfig, settings
from execution.markets import USDC_ARBITRUM, MarketResolver
from risk.risk_manager import OpenPosition, RiskDecision
from strategy.signal import Direction

logger = logging.getLogger("execution.gmx")

USDC_DECIMALS = 6


@dataclass
class OrderResult:
    submitted: bool
    debug_only: bool
    tx_hash: Optional[str]
    details: str


class GmxExecutionClient:
    def __init__(self, chain_cfg: ChainConfig, exec_cfg: ExecutionConfig):
        self.chain_cfg = chain_cfg
        self.exec_cfg = exec_cfg
        self._w3 = None
        self._wallet = None
        self._gmx_config = None
        self._market_resolver = None

    def _ensure_connected(self):
        if self._gmx_config is not None:
            return
        if self.exec_cfg.debug_mode:
            return  # no live connection needed for dry-run

        from web3 import Web3
        from eth_defi.gmx.config import GMXConfig
        from eth_defi.gmx.data import GMXMarketData
        from eth_defi.hotwallet import HotWallet

        self._w3 = Web3(Web3.HTTPProvider(self.chain_cfg.rpc_url))
        private_key = self.chain_cfg.private_key
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key  # HotWallet requires the 0x prefix
        self._wallet = HotWallet.from_private_key(private_key)
        self._wallet.sync_nonce(self._w3)

        # Chain is auto-detected from the web3 connection's chain_id;
        # there's no "chain=" kwarg on GMXConfig.
        self._gmx_config = GMXConfig(
            web3=self._w3,
            user_wallet_address=self._wallet.address,
            wallet=self._wallet,
        )
        self._market_resolver = MarketResolver(GMXMarketData(self._gmx_config))

    def _sign_and_broadcast(self, tx: dict) -> str:
        from eth_defi.hotwallet import get_tx_broadcast_data

        signed = self._wallet.sign_transaction_with_new_nonce(tx)
        raw = get_tx_broadcast_data(signed)
        tx_hash = self._w3.eth.send_raw_transaction(raw)
        return tx_hash.hex()

    def place_order(
        self,
        symbol: str,
        direction: Direction,
        decision: RiskDecision,
    ) -> OrderResult:
        collateral_usd = decision.size_usd / max(decision.leverage, 1e-9)
        summary = (
            f"{direction.value.upper()} {symbol} | size=${decision.size_usd} "
            f"leverage={decision.leverage}x collateral=${collateral_usd:.2f} "
            f"SL={decision.stop_loss_pct}% TP={decision.take_profit_pct}% "
            f"reason='{decision.reason}'"
        )

        if self.exec_cfg.debug_mode:
            logger.info("[DEBUG MODE - no order submitted] %s", summary)
            return OrderResult(submitted=False, debug_only=True, tx_hash=None, details=summary)

        self._ensure_connected()

        from eth_defi.gmx.order import IncreaseOrder

        market = self._market_resolver.resolve(symbol, settings.market.market_address_overrides.get(symbol))
        is_long = direction == Direction.LONG
        collateral_raw = int(round(collateral_usd * (10 ** USDC_DECIMALS)))

        order = IncreaseOrder(
            config=self._gmx_config,
            market_key=market.market_key,
            collateral_address=market.collateral_address,
            index_token_address=market.index_token_address,
            is_long=is_long,
        )
        result = order.create_increase_order(
            size_delta=decision.size_usd,
            initial_collateral_delta_amount=collateral_raw,
            slippage_percent=self.exec_cfg.slippage_bps / 10_000 if hasattr(self.exec_cfg, "slippage_bps") else 0.003,
        )
        tx_hash = self._sign_and_broadcast(result.transaction)
        logger.info("Submitted live order: %s | tx=%s", summary, tx_hash)
        return OrderResult(submitted=True, debug_only=False, tx_hash=tx_hash, details=summary)

    def close_position(self, position: OpenPosition, reason: str) -> OrderResult:
        """Fully closes an existing position."""
        summary = (
            f"CLOSE {position.direction.value.upper()} {position.symbol} | "
            f"notional=${position.notional_usd:.2f} reason='{reason}'"
        )

        if self.exec_cfg.debug_mode:
            logger.info("[DEBUG MODE - no close submitted] %s", summary)
            return OrderResult(submitted=False, debug_only=True, tx_hash=None, details=summary)

        self._ensure_connected()

        from eth_defi.gmx.order import DecreaseOrder

        market = self._market_resolver.resolve(position.symbol, settings.market.market_address_overrides.get(position.symbol))
        is_long = position.direction == Direction.LONG

        if position.collateral_amount_raw is None:
            raise RuntimeError(
                f"Cannot close {position.symbol}: no collateral_amount_raw on this "
                f"OpenPosition. Make sure execution/positions.py populated it from "
                f"the real position data before enabling live closes."
            )

        order = DecreaseOrder(
            config=self._gmx_config,
            market_key=market.market_key,
            collateral_address=market.collateral_address,
            index_token_address=market.index_token_address,
            is_long=is_long,
        )
        result = order.create_decrease_order(
            size_delta=position.notional_usd,  # full close: remove the entire size
            initial_collateral_delta_amount=position.collateral_amount_raw,  # and all its collateral
            slippage_percent=self.exec_cfg.slippage_bps / 10_000 if hasattr(self.exec_cfg, "slippage_bps") else 0.003,
        )
        tx_hash = self._sign_and_broadcast(result.transaction)
        logger.info("Submitted live close: %s | tx=%s", summary, tx_hash)
        return OrderResult(submitted=True, debug_only=False, tx_hash=tx_hash, details=summary)

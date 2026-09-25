"""
Resolves a symbol like "BTC/USD" to the on-chain addresses GMX's order
classes need: the market contract address, the index token address, and
the collateral token address.

Built from GMXMarketData.get_available_markets(), which returns a dict
keyed by checksummed market address, e.g.:

    {
      "0xC25c...": {
          "gmx_market_address": "0xC25c...",
          "market_symbol": "BTC",
          "index_token_address": "0x47904...",
          ...
      },
      ...
    }

GMX v2's standard markets all use USDC as collateral (confirmed against
the installed eth_defi version's position-processing code), so we use
the fixed USDC address rather than trying to derive it per-market.
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("execution.markets")

# Native USDC on Arbitrum One. If your account uses bridged USDC.e,
# swap this for 0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC.
USDC_ARBITRUM = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"


@dataclass
class MarketAddresses:
    market_key: str            # GMX market contract address
    index_token_address: str
    collateral_address: str = USDC_ARBITRUM


class MarketResolver:
    def __init__(self, market_data):
        """market_data: an eth_defi.gmx.data.GMXMarketData instance."""
        self._market_data = market_data
        self._cache = None

    def _load(self):
        if self._cache is None:
            self._cache = self._market_data.get_available_markets()
        return self._cache

    def resolve(self, symbol: str, override_address: Optional[str] = None) -> MarketAddresses:
        base = symbol.split("/")[0].upper()
        markets = self._load()

        if override_address:
            info = markets.get(override_address)
            if not info:
                raise ValueError(f"Override address '{override_address}' for '{symbol}' not found in available markets.")
            return MarketAddresses(market_key=override_address, index_token_address=info["index_token_address"])

        matches = [
            (addr, info) for addr, info in markets.items()
            if info.get("market_symbol", "").upper() == base
        ]

        if not matches:
            raise ValueError(
                f"No GMX market found for symbol '{symbol}' (looked for base token '{base}'). "
                f"Available symbols: {sorted({v.get('market_symbol') for v in markets.values()})}"
            )

        if len(matches) > 1:
            addrs = ", ".join(addr for addr, _ in matches)
            raise ValueError(
                f"Symbol '{symbol}' is ambiguous on this chain — {len(matches)} separate "
                f"GMX markets share the ticker '{base}' at addresses: {addrs}. "
                f"Refusing to guess which one to trade. Set an explicit override for this "
                f"symbol via MarketConfig.market_address_overrides in config/settings.py "
                f"(pick the address after checking each market's long/short token info)."
            )

        market_address, info = matches[0]
        return MarketAddresses(
            market_key=market_address,
            index_token_address=info["index_token_address"],
        )

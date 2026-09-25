"""
Lists every market GMX currently has available on your configured chain,
with the exact `market_symbol` string the bot's MarketResolver will match
against. Run this before adding new symbols (e.g. GOLD, SILVER, SPCX) to
config/settings.py, so you use the exact string GMX actually returns
rather than guessing between a name ("GOLD") and a ticker ("XAU").

This is read-only — no private key or signing needed.

Run with: python3 scripts/list_markets.py
"""
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings


def main():
    from web3 import Web3
    from eth_defi.gmx.config import GMXConfig
    from eth_defi.gmx.data import GMXMarketData

    w3 = Web3(Web3.HTTPProvider(settings.chain.rpc_url))
    gmx_config = GMXConfig(web3=w3, user_wallet_address=settings.chain.wallet_address or None)
    market_data = GMXMarketData(gmx_config)

    markets = market_data.get_available_markets()
    print(f"Found {len(markets)} markets on {settings.chain.chain}:\n")

    rows = sorted(
        ((info.get("market_symbol", "?"), addr) for addr, info in markets.items()),
        key=lambda r: r[0],
    )
    for symbol, addr in rows:
        print(f"  {symbol:<10} -> {addr}")

    print(
        "\nUse the left-hand symbol (before ' -> ') in config/settings.py's "
        "MarketConfig.symbols, formatted as '<SYMBOL>/USD'."
    )


if __name__ == "__main__":
    main()

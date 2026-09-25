"""
Generate a fresh Arbitrum/EVM wallet for the trading bot.

Run this once, locally, on a machine you trust:

    pip install eth-account
    python3 scripts/generate_wallet.py

It prints an address + private key + mnemonic to your terminal and
nowhere else — it makes no network calls. Copy the address and private
key into your .env (GMX_WALLET_ADDRESS / GMX_PRIVATE_KEY) and then:

  - Clear your terminal scrollback / history after copying the key.
  - Write the mnemonic down offline (paper, not a note app or screenshot)
    as your backup — anyone with it can recreate the private key.
  - Fund this wallet with ONLY the amount you're willing to risk through
    the bot: some ETH on Arbitrum for gas + execution fees, and USDC for
    collateral. Treat it as a hot wallet, not a savings account.
  - Never paste the private key into chat, a GitHub issue, a log file,
    or anywhere other than your local .env.
"""
from eth_account import Account

Account.enable_unaudited_hdwallet_features()


def main():
    account, mnemonic = Account.create_with_mnemonic()
    private_key_hex = account.key.hex()
    if not private_key_hex.startswith("0x"):
        private_key_hex = "0x" + private_key_hex  # eth_defi's HotWallet requires the 0x prefix
    print("=" * 70)
    print("NEW WALLET GENERATED — copy these now, then clear your terminal.")
    print("=" * 70)
    print(f"Address:     {account.address}")
    print(f"Private key: {private_key_hex}")
    print(f"Mnemonic:    {mnemonic}")
    print("=" * 70)
    print("Next steps:")
    print("1. Put Address into GMX_WALLET_ADDRESS in .env")
    print("2. Put Private key into GMX_PRIVATE_KEY in .env")
    print("3. Write the mnemonic down offline as backup, then clear scrollback")
    print("4. Send a small amount of ETH (gas) and USDC (collateral) on")
    print("   Arbitrum to the Address above — only what you're willing to risk")


if __name__ == "__main__":
    main()

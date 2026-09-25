# gmx_bot

A real, on-chain-capable GMX v2 trading bot for Arbitrum, built on
`web3-ethereum-defi` (`eth_defi.gmx`). Defaults to `debug_mode = True`
(dry-run) — nothing signs or broadcasts a transaction until that's
explicitly turned off in `config/settings.py`. Pairs with the
`gmx_charts` app elsewhere in this repo, which displays this bot's live
signals/regime reads/risk decisions read-only — see "Chart integration"
below.

> **This file was written from outside a full view of the codebase.**
> See `HANDOVER.md`'s "What this document is based on" section before
> trusting anything here about `config/settings.py`, `risk/risk_manager.py`,
> `indicators/rsi.py`, or `scripts/` — those files were never read while
> writing this. Everything about `main.py`, `strategy/`, `data/market_data.py`,
> and `execution/` (except `gmx_client.py`'s private-key handling, which
> also wasn't re-verified here) reflects the actual current source.

## What's here

```
apps/trading/gmx_bot/
  main.py                  the trading loop — see "How a cycle works" below
  config/
    settings.py             NOT independently reviewed for this doc — see above
  data/
    market_data.py           GMX candle fetch, freshness caching, 1w/1M resampling
  strategy/
    signal.py                 RSI-vote entry signal (5m/15m/1h) + RSI-reversion exit
    regime.py                  medium/long-term bias filter (1d/1w/1M) — see below
  risk/
    risk_manager.py            NOT reviewed for this doc — evaluates RiskDecision
                              (approved, reason, size_usd, leverage, stop_loss_pct,
                              take_profit_pct) per signal; exact sizing logic unknown here
  execution/
    positions.py               reads real open positions + account equity from GMX
    markets.py                  resolves "BTC/USD" -> GMX market/token addresses
    gmx_client.py                turns an approved decision into a signed, broadcast order
    status_writer.py             writes data/status.json for gmx_charts to read
  indicators/
    rsi.py                      NOT reviewed for this doc
  scripts/                    NOT reviewed for this doc (list_markets.py, generate_wallet.py)
  logs/, .env, .env.example, requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in RPC URL, wallet address, and (if going live) private key
```

Confirm `.env` is actually gitignored before committing anything —
`scripts/generate_wallet.py` and `execution/gmx_client.py`'s private-key
handling mean this file can hold real secrets.

### Environment variables this session added

- `LOG_LEVEL=DEBUG` — shows every symbol's RSI each cycle (via
  `signal.display()`), not just ones that crossed a threshold. Useful
  for confirming the pipeline is alive during a quiet market instead of
  wondering if something broke.
- `DRY_RUN_EQUITY_USD=1000` — overrides the real on-chain equity read
  entirely, **only while `debug_mode` is True**. Lets you exercise
  `RiskManager.evaluate()`'s sizing/scaling logic (including regime
  counter-trend scaling) without funding the wallet first. Ignored once
  `debug_mode` is False, so it can never mask a real equity-read
  failure in production.

## Running it

```bash
python3 main.py
```

Or, alongside `gmx_charts` in one command instead of two terminals:

```bash
python3 apps/trading/run_all.py
```

### How a cycle works (`main.run_cycle`)

1. Fetch real open positions + account equity from GMX.
2. **Classify regime once** for every currently-held symbol
   (`classify_position_regimes`) — shared between steps 3 and 4 so the
   same weekly/monthly candles aren't fetched twice in one cycle.
3. **Exits**: for each open position, `generate_exit_signal` checks an
   RSI-reversion exit on `exit_timeframe`, adjusted by that position's
   regime (see "Regime-aware exits" below). Closes go through
   `execution/gmx_client.py`.
4. **Entries**: for every symbol with no open position, one combined
   signal (`generate_multi_timeframe_signal` — RSI voted across
   `signal_timeframes`, filtered by `trend_timeframes`), then
   `apply_regime_filter` scales/vetoes it based on `classify_regime`'s
   read, then `RiskManager.evaluate()` decides size/leverage/SL/TP or
   rejects it. Approved decisions go to `execution_client.place_order`.
5. **Status snapshot**: `execution/status_writer.write_status` dumps
   this cycle's positions + every scanned symbol's signal/regime/risk
   decision to `data/status.json`, atomically (temp file + `os.replace`)
   so a concurrent reader never sees a half-written file.
6. Zero-signal cycles log the 3 symbols closest to crossing a threshold
   (`_log_near_misses`), so "no signals" reads as "nothing's close"
   instead of looking identical to the pipeline being broken.

## Strategy layers

**Entry (`strategy/signal.py`, `generate_multi_timeframe_signal`):**
mean-reversion RSI voted across `signal_timeframes` (default 5m/15m/1h)
— needs `min_entry_agreement` timeframes to agree before considering a
direction at all — then `trend_timeframes` (4h/1d) act as a veto/scale
filter on that decision.

**Regime (`strategy/regime.py`, new this session):** a *filter*, not a
signal generator — never turns a flat read into a trade, only
scales/vetoes an entry signal that fights the medium/long-term bias.
Votes across `regime_timeframes` (default 1d/1w/1M) against a wide
45–55 neutral band, not an oversold/overbought extreme like the entry
logic — see `HANDOVER.md` for why that distinction matters and isn't
just stylistic. A regime-aligned signal is annotated and passed through
unchanged; a counter-trend one gets `risk_scale` shrunk to 0.3x (or
vetoed to flat entirely, if `apply_regime_filter(..., veto_counter_trend=True)`).

**Regime-aware exits (`generate_exit_signal`, extended this session):**
optional `regime_bias` parameter widens the RSI exit threshold for a
regime-aligned position (more room to run) and tightens it for a
counter-trend one (exits sooner). Omitting it reproduces the exact
original behavior.

## Chart integration (`gmx_charts`)

`execution/status_writer.py` writes `data/status.json` once per cycle:
per-symbol direction, reason, entry/trend/regime RSIs, regime
votes/total, risk-manager verdict (and size/leverage/SL/TP if
approved), plus open positions and current equity. `gmx_charts`' Dash
app polls this file every 5 seconds and shows it in the sidebar/detail
panel — **read-only, one-way**. Nothing in `gmx_charts` imports from or
calls into this bot's `execution/`/`risk/`; if you're adding a chart
feature, add a field to `status.json` here and read it there, never the
reverse.

## Known gaps

- **`OpenPosition` has no `entry_price` field**, so `gmx_charts` can't
  draw an entry-price line on the chart for an open position yet. The
  chart side is fully wired and waiting — see `HANDOVER.md` for the
  exact field to add and where.
- **`risk/risk_manager.py`'s actual sizing/scaling logic was never
  reviewed** while building the regime layer or writing these docs —
  `RiskDecision`'s field names (`approved`, `reason`, `size_usd`,
  `leverage`, `stop_loss_pct`, `take_profit_pct`) are known from how
  `gmx_client.py` and `status_writer.py` consume them, but the actual
  sizing math inside `evaluate()` is not.
- **`config/settings.py`'s exact shape was never reviewed** — every
  reference to it in new code (`settings.market.regime_timeframes`,
  `settings.rsi.exit_neutral`, etc.) uses `getattr(..., default)` where
  the field is new, specifically because its existence couldn't be
  confirmed. Worth adding these as real fields once confirmed working.

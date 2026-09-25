# gmx_charts

A TensorLang app: GMX-native market data, a composite trading signal
(`indicators.tl`), a backtester, and a Dash/Plotly chart renderer with
a TradingView-style sidebar — now also a live read-only display surface
for `gmx_bot`'s signals, regime reads, and risk-manager decisions. See
`HANDOVER.md` in this same directory for full project context,
architectural decisions, and what's still open — read that first if
you're picking this up fresh.

All commands below assume your working directory is the repo root
(`tensor-lang/`), not this directory, since `tensorlang.py` and the
`cache/` it writes to both live there.

## What's here

```
apps/trading/gmx_charts/
  app.toml               metadata, entry point = indicators.tl
  indicators.tl           SMA(10/30) crossover gated by RSI(14) — pure
                           tensor arithmetic (see the big comment at
                           the top of the file for why)
  indicators_no_gate.tl   same crossover, RSI gate removed — an A/B
                           baseline for backtest.py's --entry flag
  tools/
    fetcher.py             polls GMX's REST candles/markets, normalizes
                            into 8-column bar rows; freshness-aware
                            caching (see below) and buyable/synthetic
                            market classification also live here
    tensor_store.py         plain numpy rolling (200, 8) window, one
                            file per market+period under data/
    signal_agent.py          subprocess wrapper around indicators.tl,
                            defaults to HOLD (not a random guess) if
                            the TensorLang subprocess fails
    backtest.py              slides the window across historical
                            candles, scoring signals against actual
                            next-bar returns
    chart_server.py          Dash/Plotly browser UI: sidebar, dark
                            mode, TradingView-style timeframe row,
                            gmx_bot signal/regime overlay, maintenance
                            actions — the bulk of this session's work
  data/                    durable per-market window/chart files, plus
                           market_backing.json (see below)
```

## Chart Renderer (browser UI)

```bash
pip install dash plotly pandas
cd apps/trading/gmx_charts/tools
python3 chart_server.py
```

Open `http://127.0.0.1:8050`. Or, to run this alongside `gmx_bot` with
one command instead of two terminals:

```bash
python3 apps/trading/run_all.py
```

This starts both processes, streams their logs interleaved with
`[bot]`/`[chart]` prefixes, and stops both cleanly on Ctrl+C (or if
either one crashes on its own).

### gmx_bot integration — one-way, read-only

If `gmx_bot` is running (or has run before), the sidebar shows its
signals live:

- **A colored dot** next to a symbol with an active signal — green for
  long, red for short. **Filled (●)** means the risk manager approved
  it (would have traded, funds/limits permitting); **hollow (○)** means
  the signal fired but was rejected (no equity, exposure cap, etc.).
  Hover for the reason.
- **Symbol name tint** — independent of the dot, shows the bot's
  medium/long-term *regime* read (bullish/bearish) for every scanned
  symbol, not just ones with an active entry signal. Darker shade means
  every regime timeframe agreed; lighter means just enough did.
- **"Bot live ($equity)" / "Bot offline"** label next to "Markets" —
  turns grey if `gmx_bot`'s `status.json` hasn't been refreshed in the
  last 60s, so a stopped bot can't look like a live one.
- **"Next signal →"** button jumps the selection to the next symbol
  with an active signal, scrolling it into view.
- **The resizable panel at the bottom of the sidebar** (drag the
  bottom-right corner) shows full detail for whichever symbol is
  selected: price, regime + its reasoning, the active signal, and the
  risk manager's actual verdict (size/leverage/SL/TP if approved, or
  the rejection reason if not), plus open-position PnL.

This is strictly **one-way**: `chart_server.py` only ever *reads*
`gmx_bot/data/status.json`, written once per cycle by `gmx_bot/main.py`
via `execution/status_writer.py`. Nothing in this app imports from or
calls into `gmx_bot`'s `execution/`/`risk/` — a bug in this Dash code
can't reach the wallet path. See `gmx_bot`'s own README/HANDOVER for
what's actually in that file.

### Buyable (spot-backed) vs synthetic-only assets

GMX splits assets into two real categories: ones with actual asset
liquidity locked in a GM pool (you can genuinely buy/hold/withdraw
them) and purely synthetic perp markets backed by unrelated collateral
(GOLD, SPY, QQQ, NATGAS, etc. — GMX obviously can't custody real gold
or S&P shares on-chain, so those markets are backed by ETH/USDC or
WBTC.b/USDC instead, offering price exposure only).

- Synthetic-only symbols get a faint amber background tint in the
  sidebar (both themes).
- **"Show buyable (spot-backed) only"** checkbox filters the list.
- The detail panel shows which one the selected symbol is, plus a link
  to GMX's own trading app for actual execution — this tool doesn't
  (and won't) build swap/trade execution itself, on purpose, to keep
  the read-only/execution boundary intact.

Classification comes from GMX's real `/markets` endpoint
(`fetcher.fetch_market_backing`), cached to
`data/market_backing.json` for 7 days (pool composition changes far
slower than price data). A symbol missing from that cache shows no
tint at all — treated as "unknown," never defaulted to either category.

### Chart settings → maintenance actions

Inside the "Chart settings" disclosure, below the overlay checkboxes:

- **🗑 Clear chart cache** — asks for confirmation, then deletes every
  cached `.npy` in `data/` and immediately refetches/redraws whatever's
  currently selected. Everything else goes back to fetch-on-click.
- **⬇ Load all daily data** — runs in a background thread (doesn't
  freeze the app) and walks every symbol's `1d` data sequentially
  (deliberately not parallelized — this is a manual, occasional action
  against a free API, not a hot path), showing live progress
  (`Loading daily data… 12/42`). Only backfills `1d` — the sidebar's
  price/% figures — not every timeframe for every symbol.

### Dark mode

Toggle next to "Chart settings." Persisted across reloads via the
browser's `localStorage` (a normal Dash feature for a locally-run app
like this one — unrelated to any hosted-artifact storage restrictions).
Built on CSS variables (`--bg`, `--text`, `--border`, etc., defined
once in `chart_server.py`'s `index_string`) rather than a callback
rewriting every component's `style` — important because the detail
panel is resizable via plain CSS drag, and a callback blindly
overwriting its whole `style` dict on every theme toggle would reset
your manually-dragged height back to default each time. The candlestick
chart itself needs separate handling (`THEME_PLOTLY` in
`chart_server.py`) since Plotly renders its own canvas and can't see
page CSS variables.

### Timeframes: 1m–1mo, now as buttons, and now actually refreshing

TradingView-style pill buttons replace the old dropdown. 1m through 1d
come straight from GMX; 1w and 1mo are resampled locally from cached
daily candles (GMX has no native weekly/monthly period).

**The "stale forever" bug is fixed.** `fetcher.ensure_chart_history`
used to cache a market+period's data to disk and return that same file
forever, however old — clicking a symbol you'd viewed before, or
reselecting the same timeframe, never re-fetched. `fetcher.
ensure_chart_history_fresh` now checks the cache file's age against
that period's own poll interval and, if stale, re-fetches just the
last few candles and merges them in rather than re-downloading
everything. The currently-open chart also auto-refreshes every 30s in
the background — deliberately *not* extended to the whole 40+-symbol
sidebar at once, to avoid hammering GMX for symbols nobody's currently
watching.

### % change vs. price freshness

**% change is always "vs yesterday's close," independent of the
chart's selected period** — computed by `fetcher.ensure_daily_reference`,
kept separate from whichever period button is active. **Price still
reflects the period actually being viewed** — the selected row shows a
small period badge (e.g. "AAVE `1h`") to indicate that.

### Chart overlays

Every overlay indicator registered in `tools/chart_indicators.py` — SMA
20/50, EMA 20/50, Bollinger Bands (20, 2) out of the box — plus the
current-price line and, when the bot reports an entry price for an open
position, an entry-price reference line (currently inert — see
"Known gaps" below). Adding a new overlay means one entry in
`chart_indicators.OVERLAY_INDICATORS`, nothing else. Y-axis tick
precision adapts to the asset's price magnitude automatically.

## Backtesting before you trust a live signal

```bash
cd apps/trading/gmx_charts/tools
python3 backtest.py --market ETH --period 1h --candles 2000 --stride 24 --max-windows 50
```

Start small (`--max-windows 50` is the default) and time it before running
a bigger range — each window position is a full `tensorlang.py`
subprocess (compile + CUDA launch), same cost as one live signal call,
not a cheap in-process loop.

Read the scorecard skeptically, not as a verdict — see `HANDOVER.md`'s
"Signal quality" section for the actual (unflattering) numbers from the
one real backtest run so far.

### Comparing two signal variants fairly

```bash
python3 backtest.py --market ETH --period 1h --candles 2000 \
  --stride 24 --max-windows 50 --entry indicators.tl \
  --save-snapshot /tmp/eth_1h_snapshot.npy

python3 backtest.py --market ETH --period 1h --candles 2000 \
  --stride 24 --max-windows 50 --entry indicators_no_gate.tl \
  --load-snapshot /tmp/eth_1h_snapshot.npy
```

Pins the exact same historical data across both runs — GMX's candles
endpoint only returns "the most recent N candles as of right now," so
two runs hours apart otherwise silently test different data.

## Testing it locally (needs a CUDA GPU, per app.toml's `gpus = 1`)

1. `pip install requests numpy dash plotly pandas`
2. Seed some real data:
   ```bash
   cd apps/trading/gmx_charts/tools
   python3 fetcher.py --market ETH --period 1h --once
   ```
3. Run the signal once:
   ```bash
   python3 signal_agent.py --market ETH --period 1h
   ```
   Expect `{"action": "HOLD", "raw": 0.0, "error": None}` on the very
   first run — correct behavior with a fresh, mostly-zero window.

## Known gaps, on purpose

- **Entry-price chart line is wired but inert.** `chart_server.py` will
  draw it the moment `gmx_bot`'s `status.json` reports an
  `entry_price` for an open position — but `gmx_bot`'s `OpenPosition`
  doesn't expose that field yet (needs `risk/risk_manager.py`, which
  this app never had access to). See `gmx_bot`'s HANDOVER for the exact
  next step.
- **Bulk daily-load only covers `1d`.** A "load everything for every
  timeframe" version would be far heavier (40 symbols × 8 periods) and
  wasn't built — flag it explicitly if you want it, it deserves its own
  confirmation dialog.
- **No confirmed per-market deep link into GMX's own app.** The "Open
  in GMX ↗" link goes to the general trade page, not the specific
  market — no reliable URL query-parameter scheme for that was found.
- **`fetch_markets()`'s `/tokens` schema is still unverified** (see
  HANDOVER) — separately, `/markets` (used for buyable/synthetic
  classification) turned out to be reachable and IS confirmed, so it's
  a viable alternate source for the market list itself if `/tokens`
  ever causes trouble.
- **No `if`/`elif` in `indicators.tl`** — `ast_builder.py` doesn't
  dispatch `if_statement` yet; the composite signal is arithmetic
  instead. See HANDOVER for the full explanation.

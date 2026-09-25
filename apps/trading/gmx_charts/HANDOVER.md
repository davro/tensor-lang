# HANDOVER — apps/trading/gmx_charts

Written for whichever AI assistant picks this up next. Originally built
in one sandboxed session with no GPU and a restricted network (see
"Sandbox limitations, updated" below — some of that has since changed).
A second, much longer session then integrated it with `gmx_bot` (a
separate, real, on-chain-capable trading bot) and rebuilt large parts of
`chart_server.py`. Read this whole file before making changes; several
design choices only make sense in light of things discovered later, and
at least one (see "The buyable/synthetic classification story" below)
only makes sense in light of a real bug that was caught by testing
before shipping, not by inspection.

## What this is now

A TensorLang app that turns GMX market data into a browsable chart, a
composite trading signal computed in TensorLang, a backtester — **and,
as of this session, a live read-only dashboard for `gmx_bot`**, a
separate real trading bot elsewhere in this repo. This app never
executes anything; `gmx_bot` does that, entirely independently.

```
Fetcher (fetcher.py)          — GMX REST client, candle caching/
        ↓                        freshness, market-backing classification
Tensor Store (tensor_store.py) — plain numpy, fixed (200, 8) rolling window
        ↓
indicators.tl (TensorLang)     — pure tensor arithmetic, the reference signal
        ↓                         ↓
Chart Renderer             gmx_bot (separate app, separate process)
(chart_server.py)                 ↓
        ↑←──────── status.json ───┘   (ONE-WAY: bot writes, chart only reads)
```

**The bot integration boundary is load-bearing, not incidental.**
`chart_server.py` reads `gmx_bot/data/status.json` and nothing else
from that app — no import from `gmx_bot.execution`, no import from
`gmx_bot.risk`, no function call into that codebase at all. This was a
deliberate, explicit design constraint from the start of that
integration work: a Dash/Plotly bug in this file must never be able to
reach a wallet or an order-placement call. If you're ever tempted to
"just import `gmx_bot.execution.gmx_client` real quick" for some chart
feature — don't. Add a field to `status.json` instead, via `gmx_bot`'s
`status_writer.py`, and read it here.

## Architecture decisions from the original build (still true)

**Why TensorLang only sees flattened numeric tensors, never strings or
structs; why the 8-column bar layout is a convention, not a type; why
window shifting happens in plain numpy, not TensorLang; why
`indicators.tl` uses only arithmetic instead of `if`/`elif`; why chart
data and the indicator window are separate files; why chart overlays
live in `chart_indicators.py`, not `indicators.py`; why 1w/1mo are
resampled locally instead of requested from GMX** — all unchanged from
the original build, and the original detailed reasoning for each is
preserved below unedited, since it's still exactly right.

**Why `indicators.tl` uses only arithmetic (`greater`/`less`/`mult`/
`minus`), never `if`/`elif`:** checked the actual compiler source
(`tensorlang/ast_builder.py`), not just the grammar. The grammar
defines `if_statement`, `elif_clause`, `comparison_expr` — but
`ast_builder.py`'s statement dispatch has **no case for `if_statement`
at all**. It parses, compiles to nothing, and silently does not
execute. `for_statement` and `rebind_statement` ARE fully wired to CUDA
codegen; `greater`/`less`/`equal` ARE fully wired too — that's why the
composite signal is built as pure arithmetic. **If a future session
implements `if_statement` dispatch, this is worth revisiting.**

**Why chart data and the indicator window are two separate files per
market:** `indicators.tl` needs a fixed 200-row shape (~8 days on 1h).
A chart should show months of history. `data/{market}_{period}.npy`
(200 rows, indicator input) and `data/{market}_{period}_chart.npy` (up
to `CHART_CANDLES`=5000 rows, display only) are unrelated files. Don't
collapse them back together.

**Why chart overlay indicators live in `chart_indicators.py`,
deliberately NOT `indicators.py`:** keeps an obvious naming boundary
from `indicators.tl`. Chart overlays are display-only pandas math with
zero connection to the trading signal. A chart overlay just needs to
look sensible; the trading signal needs to be provably correct.

**Why 1w/1mo are resampled locally instead of requested from GMX:**
GMX's `/prices/candles` endpoint only documents `1m, 5m, 15m, 1h, 4h,
1d` — confirmed directly from `gmx_bot/data/market_data.py`'s
`TIMEFRAME_TO_SECONDS` map (that codebase hit this as a real runtime
error — `ValueError: Unsupported timeframe: 1w` — not a guess). Rather
than gamble on an undocumented period string, both apps resample
cached `1d` candles locally instead (pandas here, hand-rolled calendar
bucketing in `gmx_bot`'s `market_data.py` — see that app's HANDOVER for
why it couldn't just reuse pandas).

## What was actually built/changed this session, and why

**Why the bot's signal is shown as a filled-vs-hollow dot, not just a
colored one:** an early version colored every active signal green/red
regardless of outcome — but most signals in `gmx_bot`'s dry-run logs
were being rejected by the risk manager (no equity yet), and a plain
colored dot made "the bot fired a signal" indistinguishable from "the
bot would actually have traded." Filled (●) = risk-manager-approved,
hollow (○) = fired but rejected. `status_writer.py` on the bot side
already tracks the full `RiskDecision`, so this was a chart-only change
once that data existed.

**Why regime tint is on the symbol NAME, separate from the dot:** the
dot only exists for symbols with an active LONG/SHORT signal this
cycle. Regime (the bot's medium/long-term bias read) is computed for
every scanned symbol regardless of direction, and the person explicitly
wanted to see bias "even for symbols with no active entry signal" — so
it's a second, independent visual channel (faint text color) rather
than folded into the dot.

**Why regime has two color tiers (full/partial), not one:** binary
bullish/bearish made 2-of-3-timeframes-agreeing look identical to
3-of-3 — no way to see conviction. `strategy/regime.py`'s
`RegimeReading` now carries `votes`/`total`; the chart shades darker
when every regime timeframe agreed, lighter when just enough did.

**Why the sidebar's `<Li>` elements are patched via `style` for the
buyable-only filter, not re-created:** this app already learned this
lesson once (see "Sandbox limitations, updated" below for how it was
finally confirmed) — Dash's pattern-matching click tracker
(`n_clicks`) resets to 0 if a component is destroyed and recreated.
`filter_market_rows` toggles `display: none`/`block` on the existing
`<Li>` elements instead of removing them, exactly like
`update_market_rows` already patches `children` in place rather than
rebuilding the list.

**Why dark mode uses CSS variables + one class toggle, not a callback
rewriting every component's `style`:** the detail panel is
user-resizable via plain CSS `resize: vertical` — dragging it sets an
explicit `height` directly in the DOM's inline style. A callback that
re-outputs a whole new `style` dict for that element on every theme
toggle would silently reset that height back to default each time.
Instead, `chart_server.py`'s `index_string` defines `--bg`, `--text`,
`--border`, etc. as CSS custom properties, components reference them in
their *initial* inline styles (`"backgroundColor": "var(--bg)"`), and
toggling theme only ever adds/removes one `theme-dark` class on the
root `<div>` — the browser re-resolves the variables, nothing Dash-side
gets rewritten. The candlestick chart is the one exception: Plotly
renders its own canvas and can't see page CSS variables, so
`THEME_PLOTLY` in `chart_server.py` keeps literal hex values in
manual sync with `THEME_CSS` — there are only two themes and they
change rarely, so this was judged not worth building a shared
single-source-of-truth for.

**Why the bulk daily-load button runs in a background thread,
sequentially, not a thread pool:** it walks 40+ symbols against a free,
rate-limitable API as a manual, occasional maintenance action — not a
hot path. A thread pool would finish faster but risks a rate-limit for
no real benefit; a background thread (vs. no threading at all) exists
purely so the button doesn't freeze the whole Dash app for however long
that takes. Progress is polled from a plain module-level dict
(`_bulk_load_state`) once a second — fine for one writer thread and one
Dash dev-server process; would need a real queue/lock if this ever runs
under a multi-process server.

**Why the "stale forever" cache bug took two sessions to actually
fix:** `ensure_chart_history`'s docstring flagged its own limitation
from the very first version of this app ("doesn't auto-refresh...e.g. a
'Refresh' button, not built yet") — but nobody hit it until real usage
(clicking between symbols over a period of hours) made it obvious as
"prices not changing." `ensure_chart_history_fresh` checks the cache
file's `mtime` against that period's own poll interval and, if stale,
fetches just the last 5 candles (`fetch_candles`, small `limit`, the
same mechanism `poll_once` already used) and merges them via
`_merge_tail` rather than re-fetching the full history. This was
actually unit-tested in this sandbox (see below) — mocked
`fetch_candles`, confirmed zero API calls when fresh, exactly one call
when stale, and confirmed the merge correctly updates a still-forming
candle in place vs. appending a genuinely new one.

### The buyable/synthetic classification story — read this before touching `fetch_market_backing`

The person asked for a way to distinguish GMX's spot-backed assets
(real asset liquidity locked in a GM pool — genuinely buyable/
withdrawable) from purely synthetic ones (GOLD, SPY, QQQ, NATGAS,
etc. — GMX has no way to custody real gold or S&P shares, so these
markets are backed by ETH/USDC or WBTC.b/USDC collateral instead).

**First attempt was wrong, and shipping it would have mislabeled
BTC — GMX's single most obviously "real" asset — as synthetic.** The
obvious approach: fetch `GET /markets` (confirmed live and reachable —
see below), compare each market's `indexToken` address to its
`longToken`/`shortToken` addresses; equal means spot-backed. Tested
against the real response before shipping (not just eyeballed): every
`BTC/USD` market variant (`[WBTC.b-WBTC.b]`, `[tBTC-tBTC]`,
`[USDG-USDG]`) has the exact same `indexToken` address, and it matches
**none** of the actual wrapped-BTC token addresses that back the first
two pools. GMX evidently tracks "the BTC price" as one abstract oracle
identifier, separate from whichever specific BTC-pegged ERC-20 happens
to be pool collateral at any given market. Raw address equality would
have called all three pools synthetic, including the two that
genuinely hold real wrapped Bitcoin.

**Fixed by parsing GMX's own display name instead of comparing
addresses.** Every market's `"name"` field already spells out its pool
composition, e.g. `"BTC/USD [WBTC.b-WBTC.b]"`, `"GOLD/USD [ETH-USDC]"`.
`fetch_market_backing` checks whether the base symbol appears
case-insensitively as a *substring* of either bracketed pool token —
"BTC" inside "WBTC.b" is exactly the signal wanted, and it also
correctly handles staked/wrapped variants like "wstETH" for ETH. Then
re-tested against a broad, real, hand-verified set of ~35 real market
entries (every TradFi symbol in this repo's watchlist plus a dozen
major crypto ones) and got every single one right: BTC/ETH/SOL/ARB/
LINK/AAVE/GMX/UNI/PENDLE/PEPE/WIF/ANIME buyable; GOLD/SILVER/NATGAS/
WTIOIL/BRENTOIL/SPY/QQQ/SPX6900/SPCX/XAUT.v2 synthetic; BNB/MORPHO/SKY
correctly synthetic too (no native token backing them in any listed
Arbitrum GMX pool as of this data).

**A symbol can have multiple pools with different backing** — e.g.
SOL/USD has SOL-USDC (buyable), WBTC.b-USDC (synthetic-for-SOL), and
USDG-USDG (synthetic-for-SOL) pools simultaneously. This treats a
symbol as buyable if ANY of its listed pools is spot-backed, since
that's enough for the real asset to be genuinely held somewhere in
GMX's liquidity — it does NOT mean every pool for that symbol is
spot-backed, which matters if `gmx_bot`'s `execution/markets.py`
`MarketResolver` ever needs this same distinction per-pool rather than
per-symbol (it currently doesn't use it at all — this classification
is chart-display-only, read from a static cache, never fed into any
trading decision).

**`/markets` turned out to be genuinely reachable, unlike everything
else GMX-related in the original sandbox** — fetched and inspected
directly in this session. Worth revisiting `fetch_markets()`'s
long-unverified `/tokens` schema using `/markets`' `"name"` field as an
alternate symbol source if `/tokens` ever causes real trouble; it
wasn't swapped over now purely to keep this change scoped to what was
actually asked for.

**Cached to `data/market_backing.json` for 7 days**, not refetched per
poll cycle like price data — pool composition changes far slower, and
new markets get listed occasionally at most. `ensure_market_backing`
returns `{}` (meaning "unknown for everything") on a fetch failure
rather than raising, so a GMX hiccup degrades to "no badges shown," not
a crashed chart server.

## Sandbox limitations, updated

The original build had **zero** live testing of anything touching
Dash, a browser, or GMX's real endpoints. This session materially
changed that:

- **Dash and Plotly are now actually installed and exercised in this
  sandbox** (`pip install dash plotly`), unlike the original build.
  Every new/changed function this session — `market_row`,
  `build_detail_panel`, `filter_market_rows`, `add_overlays`'s
  entry-price line, the bulk-load worker, the merge/staleness logic in
  `fetcher.py` — was actually called with realistic fake data and its
  output asserted against, not just compiled. This is real unit/
  integration testing of the component tree and callback logic, not a
  live browser click-through — nobody has actually clicked the buttons
  in a real browser from this side. Still worth an explicit "does this
  actually work when you click it" pass on your end for anything
  genuinely new (the maintenance buttons, the theme toggle, the period
  row) even though the underlying logic is verified.
- **`arbitrum-api.gmxinfra.io` IS reachable from this session's
  network** — a real change from the original build's confirmed 403.
  `GET /markets` was fetched directly and used to build/verify the
  buyable/synthetic classification against real data (see above).
  `/tokens` (used by `fetch_markets` for the market list itself) was
  NOT re-verified this session — that function's schema uncertainty is
  still exactly as unresolved as the original HANDOVER describes.
- **The click-tracking `<Li>`-regeneration bug** is now confirmed fixed
  at the logic level in two independent ways: the original structural
  argument (callback graph doesn't touch `n_clicks`), and this
  session's actual test of `filter_market_rows` returning `style`
  patches against a real component tree rather than regenerating
  elements. Still not confirmed by an actual mouse click in a browser.

**Practical implication for you, still true:** don't assume something
is correct just because it compiles or because a prior session said it
was fixed. Ask for the actual error output or screenshot when something
breaks, and keep stating plainly what was actually tested vs. what was
reasoned about — this pattern caught a real, materially wrong design
(the BTC address-equality bug above) before it shipped, purely by
testing against realistic data instead of trusting the first plausible
idea.

## Signal quality — the most important open finding (unchanged)

**The composite `indicators.tl` signal (SMA 10/30 crossover, gated by
RSI 14/70/30) has NOT been shown to have real edge.** 56-60% win rate
over 50 overlapping ETH 1h windows, dramatically underperforming
buy-and-hold (+38%) over the same window. An A/B test showed the RSI
gate itself is not the problem — differences were dominated by one
outlier trade. **Do not treat this signal as validated.** Note this is
entirely separate from `gmx_bot`'s actual live strategy (RSI-vote
across multiple entry timeframes + a regime filter) — the two signals
are unrelated, `gmx_bot` doesn't call `indicators.tl` at all, and this
finding says nothing about `gmx_bot`'s strategy quality one way or the
other.

## File-by-file summary

```
apps/trading/gmx_charts/
  README.md / HANDOVER.md   usage vs. why-it's-built-this-way
  app.toml                  gpus = 1 (CUDA required), entry point indicators.tl
  indicators.tl              the reference signal — SMA(10/30) x RSI(14) gate
  indicators_no_gate.tl       same crossover, gate removed — A/B baseline only
  tools/
    fetcher.py                GMX REST client: candle fetch/cache/freshness,
                              1w/1mo resampling, market-backing classification
    tensor_store.py            numpy rolling window (200,8) + chart cache paths
    signal_agent.py             subprocess wrapper around indicators.tl
    backtest.py                 slides the window across history, scores signals
    chart_indicators.py         registry of DISPLAY-ONLY overlays (SMA/EMA/BB)
    chart_server.py             Dash/Plotly UI — sidebar, dark mode, timeframe
                              row, gmx_bot overlay, maintenance actions
  data/                      *.npy chart/indicator files, status.json is
                              gmx_bot's (lives in gmx_bot/data/, not here),
                              market_backing.json (buyable/synthetic cache)
```

## What's NOT built yet

- **Execution.** This app still does not and should not place trades —
  `gmx_bot` is the only thing in this repo that does that, and even it
  runs in `debug_mode` (dry-run) by default. Don't wire real execution
  into `chart_server.py` under any circumstance; if asked, point here.
- **Entry-price chart line is chart-side-ready, bot-side-blocked** —
  see `gmx_bot`'s HANDOVER for the exact next step (`risk_manager.py`'s
  `OpenPosition` needs an `entry_price` field this app never had
  visibility into).
- **Statistically rigorous backtesting** of `indicators.tl` — explicitly
  paused, not forgotten, unrelated to `gmx_bot`'s own strategy.
- **A real Pine-Script-like indicator language** — discussed at length,
  deliberately not started; see original reasoning preserved from the
  first HANDOVER version if this comes up again.
- **Per-market GMX deep link** — "Open in GMX ↗" goes to the general
  trade page; no confirmed URL parameter scheme for a specific market
  was found.
- **Full multi-timeframe bulk load** — the bulk-load button only
  covers `1d`; a version covering every period for every symbol would
  be much heavier and wasn't built.

## Testing philosophy used throughout — please continue it

Mock what can't be run live, say plainly what's mocked vs. confirmed,
and don't claim something works end-to-end unless it was actually
exercised somewhere — even mocked/unit-level, as this session did once
Dash became installable here. The BTC classification bug above is the
clearest example yet of why: the "obviously correct" approach was
wrong, and only testing against real data caught it before it shipped.

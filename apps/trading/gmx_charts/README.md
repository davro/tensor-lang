# gmx_charts

A TensorLang app: GMX-native market data, a composite trading signal
(`indicators.tl`), a backtester, and a Dash/Plotly chart renderer with
a TradingView-style sidebar. See `HANDOVER.md` in this same directory
for full project context, architectural decisions, and what's still
open — read that first if you're picking this up fresh.

All commands below assume your working directory is the repo root
(`tensor-lang/`), not this directory, since `tensorlang.py` and the
`cache/` it writes to both live there.

## What's here

```
apps/trading/gmx_charts/
  app.toml               metadata, entry point = indicators.tl
  indicators.tl           SMA(10/30) crossover gated by RSI(14) — pure
                           tensor arithmetic, no if/elif (see the big
                           comment at the top of the file for why)
  indicators_no_gate.tl   same crossover, RSI gate removed — an A/B
                           baseline for backtest.py's --entry flag
  tools/
    fetcher.py             polls GMX's REST candles + GraphQL trade
                            volume, normalizes into 8-column bar rows
    tensor_store.py         plain numpy rolling (200, 8) window, one
                            file per market+period under data/
    signal_agent.py          subprocess wrapper around indicators.tl,
                            defaults to HOLD (not a random guess) if
                            the TensorLang subprocess fails
    backtest.py              slides the window across historical
                            candles, scoring signals against actual
                            next-bar returns
    chart_server.py          Dash/Plotly browser UI: table-aligned
                            market sidebar (price + % change), 1m
                            through 1mo timeframes, backfills on
                            demand when you click a market that hasn't
                            been fetched yet
  data/                    durable per-market window files land here
```

## Chart Renderer (browser UI)

```bash
pip install dash plotly pandas
cd apps/trading/gmx_charts/tools
python3 chart_server.py
```

Open `http://127.0.0.1:8050`. The market list on the right comes from
GMX's `/tokens` endpoint; clicking a market that hasn't been fetched
yet triggers a one-time fetch of up to 5000 candles automatically, so
there's no need to run `fetcher.py` by hand first for a new market —
expect a short pause on first click while it fetches. The selected
market is highlighted in the sidebar.

**% change is always "vs yesterday's close," independent of the chart's
selected period.** This matches how real trading UIs keep a watchlist's
daily change constant while the open chart's resolution changes — the
% figure means the same thing whether the chart is on 5m or 1w. It's
computed by `fetcher.ensure_daily_reference`, which always reads (and,
on first call, fetches) '1d' data for that market specifically, kept
separate from whatever period the dropdown is set to.

**Price still reflects the period actually being viewed** — the last
close from that period's own data, which is why the selected row shows
a small period badge next to its symbol (e.g. "AAVE `1h`"): it tells
you how fresh that specific price is, since a 5m close and a 1d close
for the same market can differ. The badge only appears on the selected
row; the % change doesn't need one, since it's the same calculation
for every row regardless of period.

**Timeframes:** 1m through 1d come straight from GMX. 1w and 1mo are
built locally by resampling cached daily candles with pandas, not sent
to GMX as period strings — GMX's own oracle-keeper doesn't document
which period values it accepts weekly/monthly, and guessing wrong
would have meant a broken timeframe instead of a working one.

**Chart settings** (the "Chart settings" disclosure above the timeframe
dropdown): a current-price line, plus every overlay indicator
registered in `tools/chart_indicators.py` — SMA 20/50, EMA 20/50, and
Bollinger Bands (20, 2) out of the box, all toggleable independently.
The checklist and the rendering are both generated FROM that registry,
not hand-listed here — adding a new overlay indicator means adding one
entry to `chart_indicators.OVERLAY_INDICATORS`, nothing else. See that
file's docstring for why it's deliberately named `chart_indicators.py`
and kept separate from `indicators.tl`: these are for display only and
have no connection to the actual BUY/SELL/HOLD trading signal. Indicator
windows are in bar units, not time units — SMA 20 on a 1h chart uses
the last 20 hourly closes; on 1d, the last 20 daily closes — matching
how most charting platforms define moving-average periods.

**Y-axis precision now adapts to the asset's price magnitude.** A
$0.10-$0.15 range used to render with Plotly's default tick spacing,
which rounds to a handful of coarse ticks and genuinely loses the price
action in between — the low-price-asset problem you ran into. The tick
format now scales decimal places to the actual price range (more
decimals for sub-$1 assets, fewer for BTC-scale prices), applied
unconditionally rather than behind a toggle, since it's a correctness
fix rather than a preference.

**The "switching timeframe forgets the selected symbol" bug is fixed.**
The sidebar's `<Li>` elements are now created once and never
recreated — only their inner content updates via a pattern-matching
callback. The previous version rebuilt the entire list (fresh
components, `n_clicks` reset to 0) every time a price changed, which
is a well-known Dash footgun for exactly this kind of click-tracking
bug. I couldn't run a real browser from the sandbox this was built in
to confirm the click flow end to end, so this is verified structurally
(callback graph validated, `n_clicks` no longer touched by the row-
content callback) rather than reproduced and re-tested directly —
worth confirming it's actually resolved on your end.

**Chart history is intentionally separate from the indicator window.**
`indicators.tl` needs a fixed 200-row shape, so `data/{market}_{period}.npy`
stays capped there on purpose. The chart has no such constraint and
caches its own, longer `data/{market}_{period}_chart.npy` (5000 candles
by default — `chart_server.CHART_CANDLES` if you want more or less).
That's still not "the full dataset since listing" for an old, heavily-
traded market — GMX's candles endpoint caps at 10000 candles per
request with no documented pagination past that, so very deep history
beyond ~5000-10000 candles isn't reachable in one call; that would need
its own paging/archival work, not implemented here. The cache also
doesn't auto-refresh once written — re-run with the file deleted, or
add a "Refresh" button, if you want newer candles appended later.

**One thing to verify on your end:** GMX's `/tokens` response schema
isn't documented with exact field names anywhere I could confirm from
the sandbox this was built in, and the live endpoint wasn't reachable
from there at all (network-restricted, confirmed 403). `fetch_markets()`
in `fetcher.py` parses defensively across a few likely shapes and falls
back to a fixed `["ETH", "BTC"]` list if none of them match — so the
app won't crash either way, but if the sidebar only ever shows those
two, your real response probably uses different field names than
guessed. `print(resp.json())` once inside `fetch_markets()` will show
you the actual shape, and it's a small fix from there.

## Backtesting before you trust a live signal

```bash
cd apps/trading/gmx_charts/tools
python3 backtest.py --market ETH --period 1h --candles 2000 --stride 24 --max-windows 50
```

Start small (`--max-windows 50` is the default) and time it before running
a bigger range — each window position is a full `tensorlang.py`
subprocess (compile + CUDA launch), same cost as one live signal call,
not a cheap in-process loop. Once you know the per-window cost, scale
`--candles`/`--stride`/`--max-windows` up deliberately.

Read the scorecard skeptically, not as a verdict:
- Overlapping windows share almost all their rows with their neighbors,
  so "50 signals" is not 50 independent trials — don't treat the win
  rate like a p-value.
- No fees, slippage, or GMX funding costs are modeled. A marginal
  result here could easily flip once those are included.
- It's one asset, one timeframe, one fixed parameter set (SMA 10/30,
  RSI 14, thresholds 30/70) — say nothing about anything you haven't
  explicitly run it against.

### Comparing two signal variants fairly

GMX's candles endpoint only returns "the most recent N candles as of
right now" — there's no documented from/to timestamp parameter — so
running the same backtest command twice, hours apart, silently tests
two different (forward-shifted) slices of history, not the same one.
That makes any before/after comparison meaningless unless the data is
pinned first:

```bash
# First run: fetches live, saves a snapshot
python3 backtest.py --market ETH --period 1h --candles 2000 \
  --stride 24 --max-windows 50 --entry indicators.tl \
  --save-snapshot /tmp/eth_1h_snapshot.npy

# Second run: reuses that exact snapshot, tests the no-gate variant
python3 backtest.py --market ETH --period 1h --candles 2000 \
  --stride 24 --max-windows 50 --entry indicators_no_gate.tl \
  --load-snapshot /tmp/eth_1h_snapshot.npy
```

Both runs now see byte-identical history, so any difference in win
rate or average return is attributable to the RSI gate itself, not to
having tested two different markets by accident.

## Testing it locally (needs a CUDA GPU, per app.toml's `gpus = 1`)

1. Install the one new dependency this app needs beyond the base repo:
   ```bash
   pip install requests numpy
   ```

2. Seed some real data (ETH perp market on Arbitrum, 1h candles):
   ```bash
   cd apps/trading/gmx_charts/tools
   python3 fetcher.py --market ETH --period 1h --once
   ```
   This should create `apps/trading/gmx_charts/data/ETH_1h.npy` and
   `cache/apps/trading/gmx_charts/indicators.tl/window.npy` at the repo
   root. Volume columns will stay at 0 / unconfirmed unless you also
   pass `--market-address <GMX ETH market contract address>` — the
   REST candle endpoint alone is enough to test the signal end to end.

3. Run the signal once:
   ```bash
   python3 signal_agent.py --market ETH --period 1h
   ```
   Expect `{"action": "HOLD", "raw": 0.0, "error": None}` on the very
   first run — with mostly-zero rows in a fresh window, neither the
   uptrend nor downtrend gate should fire. That's the correct behavior
   to see, not a bug.

4. To watch it update over time instead of a single poll:
   ```bash
   python3 fetcher.py --market ETH --period 1h --loop &
   watch -n 60 'python3 signal_agent.py --market ETH --period 1h'
   ```

## Known gaps, on purpose

- **No `if`/`elif` in indicators.tl.** The grammar has them, but
  `ast_builder.py` doesn't dispatch `if_statement` yet (checked against
  the actual compiler source, not assumed) — so the composite signal is
  written as arithmetic (`greater`/`less`/`mult`/`minus`) instead of
  branches. Worth revisiting once that's implemented — it'll read more
  clearly as explicit branches at that point.
- **`avg_loss == 0` divide-by-zero in RSI is unhandled inside
  indicators.tl on purpose** — see the comment there. Add a check in
  `signal_agent.py` before trusting a run if this bites you in
  practice.
- **Only one market+period "in flight" at a time**, matching the
  single-slot cache convention `infer.tl` uses for `board.npy`. Running
  multiple markets concurrently means separate `data/` files (already
  supported) but you'd want to parallelize `fetcher.py`/`signal_agent.py`
  invocations yourself, or extend this into a small scheduler.

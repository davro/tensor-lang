# HANDOVER — apps/trading/gmx_charts

Written for whichever AI assistant picks this up next. This app was
built incrementally, in conversation, with a real person testing each
piece on their own GPU machine after every change (this sandbox has no
GPU and a restricted network — see "Sandbox limitations" below for
what that means for how much of this has actually been verified).
Read this whole file before making changes; several early design
choices only make sense in light of things discovered later.

## What this is

A TensorLang app that turns GMX (a DeFi perps/spot exchange) market
data into: a chart you can browse in a browser, a composite trading
signal computed in TensorLang, a backtester to sanity-check that
signal, and the scaffolding to eventually place real trades on GMX.
The long-term goal (not yet built) is live GMX execution; everything
so far is data → signal → chart, deliberately in that order, with
execution pushed to last on purpose (see "What's NOT built yet").

## Architecture, and why it's shaped this way

```
Fetcher (fetcher.py)          — plain Python, hits GMX's REST + GraphQL
        ↓
Tensor Store (tensor_store.py) — plain numpy, fixed (200, 8) rolling window
        ↓
indicators.tl (TensorLang)     — pure tensor arithmetic, the actual signal
        ↓                         ↓
Chart Renderer            Execution Engine (NOT BUILT — see below)
(chart_server.py)
```

**Why TensorLang only sees flattened numeric tensors, never strings or
structs:** its grammar (see the repo's `.lark` file) has no struct,
enum, or string type outside file paths — it's a lean, fixed-shape
autograd/tensor DSL, closer to a tiny XLA-with-backward than a general
app language. Market names, file paths, HTTP calls, and all "which
market/period is this" bookkeeping live in the Python host layer.
TensorLang only ever receives a `Tensor[f64, (200, 8)]` and returns a
`Tensor[f64, (1,)]`.

**Why the 8-column layout, and why it's a *convention*, not a type:**
TensorLang has no field names, so this is enforced by comment and
convention only, duplicated across `indicators.tl`'s docstring and
`tensor_store.py`'s constants — keep those two in sync if you change
it:
```
0 timestamp | 1 open | 2 high | 3 low | 4 close
5 volume_usd | 6 volume_confirmed | 7 trade_count
```

**Why window shifting (append/replace-last-row) happens in plain numpy
in `tensor_store.py`, not in TensorLang:** an early draft did this with
`concat`/`slice` inside a `.tl` file, and it parses fine — but the
existing `apps/games/tic_tac_toe/infer.tl` convention in this repo
explicitly keeps state/data plumbing in Python and reserves TensorLang
for the actual computation. Followed that precedent here rather than
inventing a new one.

**Why `indicators.tl` uses only arithmetic (`greater`/`less`/`mult`/
`minus`), never `if`/`elif`:** checked the actual compiler source
(`tensorlang/ast_builder.py`), not just the grammar. The grammar
defines `if_statement`, `elif_clause`, `comparison_expr` — but
`ast_builder.py`'s statement dispatch (`if stmt.data == 'function_def'
... elif stmt.data == 'rebind_statement' ... elif stmt.data ==
'for_statement'`) has **no case for `if_statement` at all**. It parses,
compiles to nothing, and silently does not execute. `for_statement` and
`rebind_statement` ARE fully wired to CUDA codegen (confirmed in
`compiler.py`). `greater`/`less`/`equal` ARE fully wired too (confirmed
real CUDA kernels in `kernel_generator.py`, not stubs) — that's why the
composite signal is built as pure arithmetic: `gated_buy =
mult(is_uptrend, not_overbought)`, etc., rather than branches. **If a
future session implements `if_statement` dispatch, this is worth
revisiting** — it would read far more clearly as explicit branches.
This is a genuinely good, well-scoped first contribution back to
TensorLang itself if anyone wants to make one.

**Why chart data and the indicator window are two separate files per
market:** `indicators.tl` needs a fixed 200-row shape (~8 days on 1h).
A chart should show months of history. Early on, the Chart Renderer
mistakenly reused the 200-row indicator window and was silently capped
at 8 days — this was a real bug, caught by the person and fixed. Now:
`data/{market}_{period}.npy` (200 rows, indicator input) and
`data/{market}_{period}_chart.npy` (up to `CHART_CANDLES`=5000 rows,
display only) are unrelated files. Don't collapse them back together.

**Why chart overlay indicators (SMA/EMA/Bollinger) live in
`chart_indicators.py`, deliberately NOT named `indicators.py`:** to
keep an obvious naming boundary from `indicators.tl`. Chart overlays
are display only, computed in plain pandas, and have zero connection
to the actual trading signal. Don't be tempted to unify these — a
chart overlay just needs to look sensible; the trading signal needs to
be provably correct. Conflating them blurs a distinction worth keeping
sharp as this grows.

**Why % change in the sidebar is decoupled from the chart's selected
period:** matches how real trading platforms work (a watchlist's daily
% change stays constant regardless of what resolution the open chart
is on). `fetcher.ensure_daily_reference` always reads/fetches `1d` data
specifically, independent of `chart_server.py`'s period dropdown. Price
in the sidebar DOES reflect whatever period is currently viewed (it's
the freshest data available for that resolution) — which is why the
selected row shows a small period badge (e.g. "AAVE `1h`"): it tells
you how fresh that specific price number is, since the % change next
to it is intentionally answering a different question.

**Why 1w/1mo timeframes are resampled locally (pandas) instead of
requested from GMX directly:** GMX's oracle-keeper `/prices/candles`
endpoint's accepted `period` values aren't documented anywhere
confirmable, and this sandbox couldn't reach the live endpoint to test
empirically either. Rather than gamble on an unconfirmed API value and
ship a timeframe that silently 400s, 1w/1mo are built by resampling
already-cached `1d` candles with `pandas.resample`. Confirmed correct
against synthetic data (14 daily candles → correct weekly OHLC
aggregation).

## Signal quality — the most important open finding

**The composite signal (SMA 10/30 crossover, gated by RSI 14/70/30)
has NOT been shown to have real edge.** A real backtest against real
ETH 1h data (50 overlapping windows, ~58 days) showed:
- 56-60% win rate — not statistically distinguishable from a coin flip
  at that sample size
- Average return per signal roughly flat to slightly positive
- **+38% buy-and-hold over the identical window** — the strategy
  dramatically underperformed simply holding the asset

An A/B test (RSI gate vs. no gate, same pinned historical snapshot via
`backtest.py --save-snapshot`/`--load-snapshot`) showed the gate is
NOT the problem — win rate/return differences between the two variants
were dominated by a single outlier trade the gate happened to filter,
not a systematic effect. **This was an initial hypothesis that turned
out to be wrong when tested** — worth remembering as a demonstration
that the A/B tooling caught a wrong intuition, which is exactly what
it's for.

**Do not treat this signal as validated. Do not wire it to real GMX
execution as-is.** The person explicitly paused further signal
validation to work on the Chart Renderer instead — this is a known,
flagged, *intentionally deferred* gap, not an oversight. If asked to
build execution, push back and point here first. Reasonable next
steps, not yet started: non-overlapping/independent windows for a
cleaner statistical read, testing across a different market regime
(the tested window was one continuous rally — results may differ in
chop/sideways conditions), or trying different SMA/RSI parameters.

## Sandbox limitations — what "verified" actually means in this codebase

This was built in a sandboxed environment with **no GPU** and a
network allowlist that does NOT include GMX's endpoints
(`arbitrum-api.gmxinfra.io`, `gmx.squids.live`) or Dash's runtime needs
beyond what pip can install. Concretely, this means:

- **No TensorLang/CUDA execution was ever run in this sandbox.**
  Every real `indicators.tl` run, every backtest scorecard, every
  chart screenshot came from the person's own GPU machine. Verification
  here was: (a) checking the actual compiler source for what's
  implemented vs. grammar-only, (b) unit-testing the Python host code
  (fetcher parsing, tensor_store windowing, chart math) against
  synthetic/mocked data, (c) booting the real Dash server and pulling
  its `/_dash-layout` JSON to confirm the component tree — but NOT
  clicking through it in a real browser.
- **GMX's exact API response schemas were sometimes guessed
  defensively rather than confirmed**, specifically `/tokens`
  (`fetcher.fetch_markets`) — parses across a few plausible shapes and
  falls back to `["ETH", "BTC"]` rather than crashing, but the real
  field names were never confirmed against a live response.
- **The Dash click-tracking bug fix (sidebar `<Li>` regeneration
  resetting `n_clicks`) was diagnosed and fixed based on a known Dash
  footgun pattern, not reproduced and re-tested in a live browser.**
  It was verified structurally (callback graph validates, `n_clicks`
  is no longer touched by the content-update callback) — the person
  has not yet explicitly confirmed the bug is gone in practice.

**Practical implication for you:** if the person reports something
broken that involves GMX's actual API responses, the Dash click flow,
or CUDA/TensorLang execution, don't assume the existing code is correct
just because it's "already tested" — large parts of it were only
testable at the mocked/structural level from here. Ask for the actual
error output and iterate the way this conversation did throughout:
change, explain what you verified and what you couldn't, ship, wait for
real confirmation, fix forward.

## File-by-file summary

```
apps/trading/gmx_charts/
  README.md              user-facing usage instructions (this file's
                          sibling — read that too, it's the "how do I
                          run this" doc; this file is "why is it built
                          this way")
  HANDOVER.md             this file
  app.toml                gpus = 1 (CUDA required), entry point indicators.tl
  indicators.tl            the live trading signal — SMA(10/30) x RSI(14) gate,
                          pure tensor arithmetic (see "if/elif" note above)
  indicators_no_gate.tl    same crossover, gate removed — A/B baseline only
  tools/
    fetcher.py             GMX REST/GraphQL client + normalization;
                          fetch_markets() is the unverified-schema one
    tensor_store.py         numpy rolling window (200,8) for the signal,
                          separate chart_data_path for the chart's own cache
    signal_agent.py          subprocess wrapper around indicators.tl,
                          degrades to HOLD (never a random guess) on failure
    backtest.py              slides the window across history, scores signals;
                          --save-snapshot/--load-snapshot for apples-to-apples A/B
    chart_indicators.py      registry of DISPLAY-ONLY overlays (SMA/EMA/BB) —
                          add one dict entry here to add a new chart overlay
    chart_server.py          Dash/Plotly browser UI — sidebar, settings panel,
                          candlestick + overlays
  data/                    *.npy files: {market}_{period}.npy (indicator,
                          200 rows) and {market}_{period}_chart.npy (chart,
                          up to 5000 rows) — DO NOT merge these back together
```

## What's NOT built yet (in the original layering: data → chart → signal → execution)

- **Execution Engine.** Nothing talks to GMX's smart contracts to place
  a trade. This was always meant to come last, behind a paper-trading
  mode, and that's still the right call — especially given the signal
  quality finding above. Don't build this until the signal question is
  resolved, or at minimum, flag loudly that it's unvalidated if asked
  to proceed anyway.
- **Statistically rigorous backtesting** (non-overlapping windows,
  multiple market regimes, fees/slippage/funding modeled). Explicitly
  paused, not forgotten.
- **A real Pine-Script-like indicator language.** Discussed at length
  and deliberately NOT started — see reasoning below. The chosen middle
  path (`chart_indicators.py`'s registry) was built instead, but only
  covers chart-display overlays, not user-authorable *trading signals*.
  If asked to revisit this: the conclusion was that building a second
  full language (with per-bar state semantics TensorLang's fixed-shape
  model doesn't naturally support) is a large, standalone project — not
  something to bolt on alongside other work. Worth scoping seriously on
  its own if it comes up again, not worth starting casually.
- **Volume bars, drawing tools, multi-pane indicators (e.g. RSI as a
  separate subplot below the price chart)** — mentioned as options,
  not chosen yet, still open.
- **A "Refresh" button for the chart cache** — right now
  `data/{market}_{period}_chart.npy` never auto-updates once written;
  the person has to delete the file to get newer candles.

## Testing philosophy used throughout — please continue it

Every change in this app's history was verified as concretely as this
sandbox allowed *before* being handed back, and every limitation on
that verification was stated explicitly rather than glossed over
(e.g. "I couldn't test the real click flow, here's what I did check
instead"). Several real bugs were caught this way before the person
ever saw them (a buy-and-hold benchmark computed over the wrong date
range, a backtest silently overwriting live data, a missing CLI flag).
Keep doing this: mock what can't be run live, say plainly what's
mocked vs. confirmed, and don't claim something works end-to-end
unless it was actually run end-to-end somewhere.

# HANDOVER — apps/trading/gmx_bot

Written for whichever AI assistant picks this up next, by an assistant
that helped build the regime layer, the `gmx_charts` integration, the
1w/1M timeframe support, and several bug fixes/features — but was never
shown the whole codebase. **Read "What this document is based on"
first.** Getting the boundary of actual vs. assumed knowledge wrong
here is exactly the kind of mistake that causes silent breakage later.

## What this document is based on

Files actually read in full, source-verified, across the session(s)
that produced this document:

- `main.py` (read multiple times, across several rounds of edits)
- `strategy/signal.py` (read in full, extended with `regime_bias`)
- `strategy/regime.py` (written from scratch this session — new file)
- `execution/positions.py` (read in full, extended with the dry-run
  equity override)
- `execution/markets.py` (read in full, unmodified)
- `execution/gmx_client.py` (read in full, unmodified)
- `execution/status_writer.py` (written from scratch this session — new
  file, extended twice)
- `data/market_data.py` (read in full, extended with 1w/1M resampling)

Files **never seen**, referenced only through how the files above
import/call them:

- `config/settings.py` — every attribute referenced from new code
  (`settings.market.regime_timeframes`, `settings.rsi.exit_neutral`,
  `settings.execution.debug_mode`, etc.) is either confirmed by
  existing usage in the files above, or accessed defensively via
  `getattr(settings.market, "regime_timeframes", [...])` specifically
  *because* it's a new field whose existence in the real file couldn't
  be confirmed.
- `risk/risk_manager.py` — `RiskDecision`'s fields (`approved`,
  `reason`, `size_usd`, `leverage`, `stop_loss_pct`, `take_profit_pct`)
  and `OpenPosition`'s fields (`symbol`, `direction`, `notional_usd`,
  `unrealized_pnl_usd`, `market_address`, `collateral_amount_raw`) are
  known only from how `positions.py`, `gmx_client.py`, and
  `status_writer.py` construct/consume them — **the actual
  `RiskManager.evaluate()` sizing/scaling algorithm has never been
  read.** This is why the entry-price chart feature stalled: adding a
  field to `OpenPosition` without seeing its real dataclass definition
  risked breaking something invisible from here.
- `indicators/rsi.py` — `compute_rsi(closes, period).latest` is the
  only interface used; the actual RSI math was never inspected.
- `scripts/list_markets.py`, `scripts/generate_wallet.py` — never
  opened.
- `requirements.txt`, any pre-existing `README.md` — never seen. The
  README shipped alongside this HANDOVER is a fresh document, not an
  edit of whatever existed before; if one already existed with content
  worth preserving, merge by hand rather than assuming this replaces it
  cleanly.

**If you're the next assistant here: don't extend this false
confidence.** When you do get to read `config/settings.py` or
`risk/risk_manager.py`, update this list and fix any `getattr` fallback
that turns out to have guessed a wrong default.

## Why the regime layer is a separate module, not folded into `trend_timeframes`

`signal.py`'s existing veto/scale logic (`trend_timeframes`, default
4h/1d) is written for **mean-reversion** RSI — it looks for genuine
extremes (oversold/overbought) and assumes price reverts from them.
That's the right model for entry timeframes. It is the **wrong** model
for weekly/monthly RSI on a real trending asset: RSI there can sit at
60-65 for weeks without ever giving a classic "overbought" reading —
treating that the same way just means fighting a real trend under the
label "risk management."

`strategy/regime.py` asks a different, simpler question instead: "which
side of neutral (50) is this asset structurally living on," via a vote
across `regime_timeframes` against a wide 45–55 band, not an extreme.
It's a **filter**, never a signal generator: a FLAT entry signal stays
FLAT no matter what the regime says; only a LONG/SHORT signal gets
scaled or vetoed based on alignment. This is why `apply_regime_filter`
takes an already-computed `MultiTimeframeSignal` and returns the same
shape back, rather than being wired into `generate_multi_timeframe_signal`
itself — keeping it a strictly additive layer that `main.py` calls
right after entry-signal generation, with the raw `RegimeReading` kept
alongside the filtered signal (not just folded into its `reason`
string) so `status_writer.py`/`gmx_charts` can display it as its own
field.

**Regime timeframes are deliberately NOT reused from
`trend_timeframes`** (separate `regime_timeframes` setting,
`getattr`-defaulted to `["1d", "1w", "1M"]`) — 4h/1d's job is the
existing entry veto; 1d/1w/1M's job is the new structural-bias read.
1d appears in both lists on purpose; that's fine, they're evaluated
independently for different purposes.

**Candle-limit overrides for 1w/1M** (`CANDLE_LIMIT_OVERRIDES` in
`regime.py`): the blanket `max(100, rsi_cfg.period * 3)` floor used
elsewhere in this codebase is fine for cheap, native-GMX-period
timeframes. Applied to 1w/1M (see below — these are *resampled*, not
native), it would mean requesting 100 monthly bars = ~8 years of daily
candles, refetched every cycle, for every symbol. Capped to 60
weekly/36 monthly bars instead — still generous for a stable RSI(14)
read, not needlessly enormous.

**Why position regime is computed once per cycle in `main.py`
(`classify_position_regimes`), not inside `handle_exits` or
`handle_entries` separately:** both regime-aware exits and the chart's
display of held-position regime need the same data for the same
symbols in the same cycle. Computing it once and threading it into both
functions avoids fetching the same weekly/monthly candles for the same
symbol twice.

## Why 1w/1M are resampled from `1d` in `market_data.py`, and the exact bug that made this necessary

GMX's `/prices/candles` endpoint has no native weekly/monthly period at
all — confirmed by hitting it directly and getting `ValueError:
Unsupported timeframe: 1w` in a real run, not by reading undocumented
API docs. `market_data.py`'s `_get_synthetic_candles`/
`_resample_daily_candles` build real calendar-week (ISO, Monday-start)
and calendar-month bars from cached `1d` candles by hand — standard
OHLCV resampling (first open, last close, min/max high/low, summed
volume), done manually rather than via pandas since this file has no
pandas dependency otherwise. **This was actually unit-tested against
known dates before shipping** (10 days of synthetic daily candles
starting on a known Thursday → correctly produced 11 weekly bars
including the partial edge week, and 3 clean calendar-month bars) — not
just eyeballed. `MAX_SYNTHETIC_LOOKBACK_DAYS` (10 years) caps the
underlying daily fetch regardless of how large a `limit` a caller
requests, so a careless `limit=500` for `"1M"` can't trigger a
multi-decade daily candle request.

## Why regime-aware exits use +10/-5 RSI-point shifts, not a config value

`generate_exit_signal`'s `regime_bias` parameter widens
(regime-aligned) or tightens (counter-trend) `exit_neutral` by a fixed
10 or 5 points rather than a new settings field — this is a first,
reasonable-but-unvalidated choice, explicitly not backtested against
real data. If you're picking this up: this is a good candidate for
`config/settings.py` fields once its effect has actually been observed
in practice, not before.

## Why `status_writer.py` writes atomically and is one-way

`os.replace` after writing to a temp file (`tempfile.mkstemp`) so
`gmx_charts`, polling every 5 seconds from a separate process, can
never read a half-written JSON document. The write happens once, at
the end of `run_cycle`, after everything for that cycle is already
computed — deliberately just serialization, no new computation. This
file is the *entire* interface to `gmx_charts`: that app imports
nothing else from this one, calls no function here, and this bot
imports nothing from `gmx_charts` either. Keep it that way — see
`gmx_charts`' own HANDOVER for why this boundary is treated as a hard
safety constraint, not a style preference.

## Why `DRY_RUN_EQUITY_USD` is gated by `debug_mode`, not a standalone env var

Every entry signal was dying on "$0.00 equity" in dry-run, making it
impossible to actually exercise `RiskManager.evaluate()`'s sizing math
(untested, since that file was never read — see above). The override
lives in `positions.py`'s `get_account_equity_usd()`, checked only
`if settings.execution.debug_mode:` — so it's structurally impossible
for this override to mask a real equity-read failure once `debug_mode`
is turned off for live trading. This was a deliberate ordering choice:
check `debug_mode` first, then look at the env var, not the reverse.

## Known gaps

- **`OpenPosition.entry_price` doesn't exist.** `gmx_charts`' entry-
  price chart line is fully built and waiting on the chart side
  (`positions[symbol].entry_price` in `status.json`, read defensively
  via `getattr(p, "entry_price", None)` in `status_writer.py` so it'll
  start flowing the moment the field exists) — this is the single
  clearest next step for whoever next has `risk_manager.py` open.
  Before adding it: run `PositionTracker.debug_dump_raw_positions()`
  (already exists, built for exactly this kind of investigation) to
  confirm what GMX's raw position dict actually calls the entry/average
  price field — don't guess a key name the way an earlier, unrelated
  bug in this same file already got bitten by drifted field names once
  (see `positions.py`'s own docstring).
- **`RiskManager.evaluate()`'s real sizing logic is unverified from
  here** — the regime counter-trend scaling (`risk_scale`) is threaded
  all the way to `RiskDecision`, but whether/how `evaluate()` actually
  uses `MultiTimeframeSignal.risk_scale` in its size calculation was
  never confirmed by reading that function.
- **Regime-aware exit shift values (10/5 RSI points) are unvalidated**
  — see above.
- **`config/settings.py` should gain real fields** for everything
  currently `getattr`-defaulted once its actual structure is confirmed:
  `market.regime_timeframes`, and the regime-filter tuning constants
  currently hardcoded in `regime.py` (`DEFAULT_BULLISH_RSI`,
  `DEFAULT_BEARISH_RSI`, `DEFAULT_REGIME_MIN_AGREEMENT`,
  `DEFAULT_COUNTER_TREND_SCALE`).

"""
apps/trading/gmx_charts/tools/chart_server.py

TradingView-style browser UI: a clickable market list on the right, a
candlestick chart on the left. Pure host-side Python + Dash/Plotly —
same layering as the rest of this app, the Chart Renderer only ever
reads from the Tensor Store; TensorLang is not involved here at all.

Run:
    pip install dash
    python3 chart_server.py
    -> open http://127.0.0.1:8050

New dependency beyond the rest of this app: dash (pulls in Flask).
"""
import json
import logging
import sys
import threading
import time
from pathlib import Path

import dash
from dash import Input, Output, State, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_indicators  # noqa: E402
import fetcher  # noqa: E402
import tensor_store  # noqa: E402

logger = logging.getLogger("chart_server")

# 1w/1mo aren't in this list because they're not sent to GMX directly —
# see RESAMPLE_RULES below. GMX's own accepted period strings for its
# oracle-keeper candles endpoint aren't documented anywhere confirmable
# from the sandbox this was built in, so these two are built locally
# instead of gambling on an unverified API value.
PERIODS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"]
DEFAULT_PERIOD = "1h"
FALLBACK_MARKETS = ["ETH", "BTC"]  # used only if fetch_markets() fails
CHART_CANDLES = 5000  # ~208 days at 1h — see fetcher.GMX_CANDLE_LIMIT for the hard ceiling

# apps/trading/gmx_charts/tools/chart_server.py -> parents[2] is apps/trading,
# sibling app's status snapshot from execution/status_writer.py. Read-only:
# nothing in this file ever writes here or imports from gmx_bot.
BOT_STATUS_PATH = Path(__file__).resolve().parents[2] / "gmx_bot" / "data" / "status.json"
BOT_STATUS_POLL_MS = 5000
# If status.json's generated_at is older than this, treat the bot as
# offline rather than trusting stale signals. gmx_bot's own poll
# interval is configurable (config/settings.py), so this is deliberately
# a generous multiple of a typical cycle rather than a tight match to
# any one setting — the goal is "the bot clearly isn't running anymore",
# not flagging one slow cycle.
BOT_STALE_AFTER_SECONDS = 60


def load_bot_status() -> dict:
    """Best-effort read of the bot's status.json. Returns {} if the bot
    isn't running yet, hasn't written a cycle, or the file is mid-write
    (write_status uses os.replace, so this should be rare) — the chart
    just shows no bot overlay in that case rather than erroring."""
    try:
        with open(BOT_STATUS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# period -> (real GMX period to fetch, pandas resample rule). Built by
# resampling already-cached daily candles rather than asking GMX for a
# period string that may not exist.
RESAMPLE_RULES = {"1w": ("1d", "W"), "1mo": ("1d", "ME")}

CHART_OPTIONS = [{"label": "Current price line", "value": "price_line"}] + [
    {"label": spec["label"], "value": key} for key, spec in chart_indicators.OVERLAY_INDICATORS.items()
]
DEFAULT_CHART_OPTIONS = ["price_line"]  # everything else off by default to keep a first-look chart uncluttered


def y_axis_tickformat(bars: np.ndarray) -> str:
    """d3-format string sized to the asset's actual price magnitude —
    same idea as format_price, applied to the chart's y-axis instead of
    the sidebar. Plotly's default tick formatting on a $0.10-$0.15
    range rounds to a handful of coarse ticks, which is exactly the
    'price action is lost in the range' problem — fixing that means
    giving it enough decimal places for the actual scale, not just
    picking a fixed number of decimals for every asset.
    """
    max_price = float(np.max(bars[:, tensor_store.COL_HIGH])) if len(bars) else 0.0
    if max_price >= 1000:
        return ",.2f"
    elif max_price >= 1:
        return ",.4f"
    elif max_price >= 0.01:
        return ".4f"
    else:
        return ".8f"


def add_overlays(fig: go.Figure, bars: np.ndarray, options: set, entry_price: float = None) -> None:
    """Adds the optional current-price line, plus whichever overlay
    indicators from chart_indicators.OVERLAY_INDICATORS are toggled on.
    Adding a new indicator to that registry needs no change here —
    this just iterates it generically.

    entry_price, when given, draws a second reference line for gmx_bot's
    open-position entry price (distinct color/dash from the current-
    price line) — sourced from status.json's positions[symbol].entry_price.
    That field doesn't exist on the bot side yet (OpenPosition has no
    entry price recorded), so this stays inert (no line drawn) until
    it's added there; see status_writer.py's comment on the same field.
    """
    if len(bars) == 0:
        return
    closes = bars[:, tensor_store.COL_CLOSE]
    timestamps = pd.to_datetime(bars[:, tensor_store.COL_TIMESTAMP], unit="s")

    for key, spec in chart_indicators.OVERLAY_INDICATORS.items():
        if key not in options or len(closes) < spec["min_bars"]:
            continue
        for trace_name, values in spec["compute"](closes).items():
            fig.add_trace(go.Scatter(
                x=timestamps, y=values, mode="lines", name=trace_name,
                line=dict(width=1, color=spec["colors"].get(trace_name),
                          dash=spec["dash"].get(trace_name)),
            ))

    if "price_line" in options:
        last_close = float(closes[-1])
        fig.add_hline(
            y=last_close, line_dash="dash", line_color="#888",
            annotation_text=format_price(last_close), annotation_position="top left",
        )

    if entry_price is not None:
        fig.add_hline(
            y=float(entry_price), line_dash="dot", line_color="#3f6fb3", line_width=2,
            annotation_text=f"entry {format_price(float(entry_price))}", annotation_position="bottom left",
        )


def get_market_list() -> list:
    """Fetched once at process start, not per page load — GMX's token
    list doesn't change minute to minute. Falls back to a small fixed
    list rather than crashing the whole server if fetch_markets()'s
    still-unverified parsing doesn't match your real response; see the
    comment on fetcher.fetch_markets for how to fix that quickly.
    """
    try:
        markets = fetcher.fetch_markets()
        symbols = sorted({m["symbol"] for m in markets if m["symbol"]})
        return symbols or FALLBACK_MARKETS
    except Exception as e:
        print(f"[chart_server] couldn't fetch market list, falling back to {FALLBACK_MARKETS}: {e}")
        return FALLBACK_MARKETS


def format_price(price: float) -> str:
    """Adaptive decimal places, same spirit as most exchange UIs — a
    fixed 2 or 4 decimals badly misrepresents either a $100k BTC or a
    $0.00001 microcap in the same list.
    """
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    else:
        return f"${price:.8f}"


def resample_bars(bars: np.ndarray, rule: str) -> np.ndarray:
    """Builds weekly/monthly OHLC candles from already-fetched daily
    bars — see the PERIODS comment for why this happens locally
    instead of asking GMX for '1w'/'1mo' directly.
    """
    if len(bars) == 0:
        return bars
    df = pd.DataFrame({
        "timestamp": bars[:, tensor_store.COL_TIMESTAMP],
        "open": bars[:, tensor_store.COL_OPEN],
        "high": bars[:, tensor_store.COL_HIGH],
        "low": bars[:, tensor_store.COL_LOW],
        "close": bars[:, tensor_store.COL_CLOSE],
    })
    df.index = pd.to_datetime(df["timestamp"], unit="s")
    resampled = df.resample(rule).agg(
        {"timestamp": "first", "open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    out = np.zeros((len(resampled), tensor_store.N_FEATURES))
    out[:, tensor_store.COL_TIMESTAMP] = resampled["timestamp"].values
    out[:, tensor_store.COL_OPEN] = resampled["open"].values
    out[:, tensor_store.COL_HIGH] = resampled["high"].values
    out[:, tensor_store.COL_LOW] = resampled["low"].values
    out[:, tensor_store.COL_CLOSE] = resampled["close"].values
    return out


def get_bars(market: str, period: str) -> np.ndarray:
    """Single entry point for 'get me this market/period as bars' —
    routes 1w/1mo through the local resampler, everything else through
    the freshness-aware GMX-backed chart cache (see
    fetcher.ensure_chart_history_fresh — the plain ensure_chart_history
    never refreshes an existing file, which was why reselecting a
    symbol or timeframe kept showing the same stale bars forever).
    """
    if period in RESAMPLE_RULES:
        base_period, rule = RESAMPLE_RULES[period]
        base_bars = fetcher.ensure_chart_history_fresh(market, base_period, n_candles=CHART_CANDLES)
        return resample_bars(base_bars, rule)
    return fetcher.ensure_chart_history_fresh(market, period, n_candles=CHART_CANDLES)


# Mirrors THEME_CSS's :root / body.theme-dark values. Plotly renders
# its own canvas and doesn't see page CSS variables, so the figure's
# colors need literal hex here — kept in sync with THEME_CSS by hand
# since there are only two themes and they change rarely.
THEME_PLOTLY = {
    "light": {"template": "plotly_white", "bg": "#ffffff", "grid": "#e5e5e5"},
    "dark": {"template": "plotly_dark", "bg": "#1a1a1a", "grid": "#333333"},
}


def build_figure_and_data(market: str, period: str, options: set = frozenset(),
                           entry_price: float = None, theme: str = "light"):
    """Renders the candlestick chart from whatever period is selected.
    Price is the last close from that SAME period's data (freshest
    available for the resolution actually being viewed). % change is
    deliberately NOT computed from that period's data — it always
    comes from fetcher.ensure_daily_reference, so every row in the
    sidebar means "change since yesterday's close" regardless of
    whether the chart is on 5m or 1w, matching how real trading UIs
    keep a watchlist's daily change independent of the open chart's
    resolution. `period` is carried in the returned dict purely so the
    sidebar can show which resolution the displayed PRICE reflects —
    see market_row's period badge, shown only for the selected row.
    `options` (a set of CHART_OPTIONS values) controls the current-
    price line and SMA overlays; the y-axis tick format adapts to the
    asset's price magnitude unconditionally, not behind a toggle — it's
    a correctness fix, not a preference.
    Returns (figure, {"price", "change_pct", "period"} | None).
    """
    theme_colors = THEME_PLOTLY.get(theme, THEME_PLOTLY["light"])
    try:
        bars = get_bars(market, period)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Couldn't load {market} {period}: {e}", template=theme_colors["template"])
        return fig, None

    if len(bars) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{market} {period}: no data yet", template=theme_colors["template"])
        return fig, None

    fig = go.Figure(data=[go.Candlestick(
        x=pd.to_datetime(bars[:, tensor_store.COL_TIMESTAMP], unit="s"),
        open=bars[:, tensor_store.COL_OPEN],
        high=bars[:, tensor_store.COL_HIGH],
        low=bars[:, tensor_store.COL_LOW],
        close=bars[:, tensor_store.COL_CLOSE],
        name=market,
    )])
    add_overlays(fig, bars, options, entry_price=entry_price)
    fig.update_layout(
        title=f"{market} / USD — {period}",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=20),
        template=theme_colors["template"],
        paper_bgcolor=theme_colors["bg"],
        plot_bgcolor=theme_colors["bg"],
    )
    fig.update_yaxes(tickformat=y_axis_tickformat(bars), gridcolor=theme_colors["grid"])
    fig.update_xaxes(gridcolor=theme_colors["grid"])

    daily = fetcher.ensure_daily_reference(market)
    data = {
        "price": float(bars[-1, tensor_store.COL_CLOSE]),
        "change_pct": daily["change_pct"] if daily else None,
        "period": period,
    }
    return fig, data


def market_row(symbol: str, entry: dict, is_selected: bool, bot_signal: dict = None,
                has_position: bool = False, bot_stale: bool = False) -> html.Div:
    """One sidebar row: symbol (left, flexible), price (right-aligned,
    fixed width), daily % change (right-aligned, fixed width, color-
    coded). Fixed-width columns are what actually gives the table-like
    alignment regardless of symbol length — text-only padding can't do
    that once symbol lengths vary (0G vs APE_deprecated). The selected
    row also gets a small badge showing which chart period its PRICE
    reflects — the % change itself always means "vs yesterday", so the
    badge is only about the price's freshness, not the % figure.

    bot_signal/has_position come from gmx_bot's status.json (see
    load_bot_status): a colored dot for the bot's current long/short
    read on this symbol (hover for its reason string), and an "[open]"
    tag if the bot currently holds a position in it. Purely a read-only
    display of what the bot already decided — nothing here can place,
    close, or affect a trade.

    bot_stale is true when status.json hasn't been refreshed recently
    (see BOT_STALE_AFTER_SECONDS) — the dot is greyed out rather than
    hidden, since the direction is still informative, but its color
    coding (which implies "current") would otherwise be misleading.
    """
    price_str, change_str, change_color, period_badge = "", "", "inherit", None
    if entry:
        price_str = format_price(entry["price"])
        change = entry.get("change_pct")
        if change is not None:
            sign = "+" if change >= 0 else ""
            change_str = f"{sign}{change:.2f}%"
            change_color = "#0a7d32" if change >= 0 else "#c92a2a"
        if is_selected:
            period_badge = entry.get("period")

    badges = []
    if bot_signal and bot_signal.get("direction") in ("long", "short"):
        reason = bot_signal.get("reason", "")
        approved = bot_signal.get("risk_approved", False)
        # Filled dot (\u25cf) = risk manager approved this signal (would
        # have traded, funds/limits permitting). Hollow dot (\u25cb) =
        # the signal fired but was rejected (e.g. no equity, exposure
        # cap, correlation limit) — still worth seeing, but not the same
        # as "the bot would actually act on this."
        dot_char = "\u25cf" if approved else "\u25cb"
        if bot_stale:
            dot_color = "#aaaaaa"
            tooltip = f"(stale — bot offline) {reason}"
        else:
            dot_color = "#0a7d32" if bot_signal["direction"] == "long" else "#c92a2a"
            tooltip = f"{reason} \u2014 {'approved' if approved else 'rejected by risk manager'}"
        badges.append(html.Span(
            dot_char, title=tooltip,
            style={"marginLeft": "6px", "color": dot_color, "fontSize": "1.3em",
                   "verticalAlign": "middle", "flexShrink": 0},
        ))

    # Regime is independent of whether there's an active entry signal —
    # bot_signal (and its regime field) is present for every scanned,
    # non-open symbol each cycle regardless of direction, so this fires
    # even when the dot above doesn't. Faint text tint rather than a bold
    # color/background so it reads as background context, not an alert.
    # Two shades per direction: "full" (every regime timeframe agreed)
    # is more saturated than "partial" (just enough votes to clear
    # min_agreement) — conviction, not just direction.
    name_style = {
        "overflow": "hidden", "textOverflow": "ellipsis",
        "whiteSpace": "nowrap", "minWidth": 0,
    }
    name_title = None
    if bot_signal and not bot_stale:
        regime = bot_signal.get("regime")
        votes = bot_signal.get("regime_votes")
        total = bot_signal.get("regime_total")
        tier = "full" if (votes is not None and total and votes >= total) else "partial"
        regime_color = REGIME_COLOR_TIERS.get(regime, {}).get(tier)
        if regime_color:
            name_style["color"] = regime_color
        if bot_signal.get("regime_reason"):
            name_title = f"Regime: {regime} ({votes}/{total}) \u2014 {bot_signal['regime_reason']}"

    if has_position:
        badges.append(html.Span(
            "[open]",
            style={"marginLeft": "4px", "fontSize": "0.7em", "color": "#555", "flexShrink": 0},
        ))
    if period_badge:
        badges.append(html.Span(
            period_badge,
            style={
                "marginLeft": "6px", "fontSize": "0.7em", "color": "#555",
                "backgroundColor": "#dde3ea", "borderRadius": "3px", "padding": "1px 4px",
                "flexShrink": 0,
            },
        ))

    # Buyable (spot-backed) vs synthetic-only — see fetcher.fetch_market_backing.
    # Independent of bot_signal entirely: MARKET_BACKING is static GMX
    # market metadata, present for every symbol GMX lists regardless of
    # whether the bot has scanned it this cycle. A symbol missing from
    # MARKET_BACKING (endpoint was unreachable, or it's a token not in
    # any GM pool) gets no tint at all rather than defaulting to either
    # category.
    is_buyable = MARKET_BACKING.get(symbol)
    backing_note = None
    row_bg = "transparent"
    if is_buyable is False:
        row_bg = "var(--synthetic-bg)"
        backing_note = f"Synthetic-only: no real {symbol} held in any GMX pool backing this market \u2014 perp price exposure only."
    elif is_buyable is True:
        backing_note = f"Spot-backed: at least one GMX pool genuinely holds {symbol} (or a token pegged to it)."
    if backing_note:
        name_title = f"{name_title}\n{backing_note}" if name_title else backing_note

    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "width": "100%",
            "backgroundColor": "var(--selected-bg)" if is_selected else row_bg,
            "fontWeight": "600" if is_selected else "normal",
            "padding": "2px 4px",
            "borderRadius": "4px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "flex": "1 1 auto", "minWidth": 0},
                children=[
                    html.Span(symbol, title=name_title, style=name_style),
                    *badges,
                ],
            ),
            html.Span(price_str, style={"flex": "0 0 90px", "textAlign": "right"}),
            html.Span(change_str, style={"flex": "0 0 60px", "textAlign": "right", "color": change_color}),
        ],
    )


MARKET_LIST = get_market_list()
# {symbol: True (spot-backed/buyable) | False (synthetic-only)} — see
# fetcher.fetch_market_backing for the classification logic. A symbol
# missing from this dict means "unknown" (e.g. GMX's /markets endpoint
# was unreachable at startup), not "synthetic" — market_row treats a
# missing key as no badge/tint at all, deliberately not defaulting to
# either category when the data just isn't there.
MARKET_BACKING = fetcher.ensure_market_backing()

# Background bulk-load state, read/written from two different Dash
# callbacks (the button that starts it, and the interval that polls
# it) plus the worker thread itself. A plain module-level dict is fine
# here — single Dash dev-server process, single writer thread at a
# time (start_bulk_load refuses to launch a second one while
# _bulk_load_state["running"] is True), and the values involved
# (an int counter, a bool, a dict of plain data) don't need anything
# fancier than the GIL already gives us for this kind of "one thread
# writes, another polls" progress reporting.
_bulk_load_state = {"running": False, "done": 0, "total": 0, "results": {}}


def _bulk_load_daily_worker():
    """Runs in a background thread so the bulk-load button doesn't
    freeze the whole Dash app for however long it takes to walk every
    symbol (deliberately sequential, not parallelized — this is a
    manual, occasional maintenance action against a free API, not a
    hot path; hammering GMX with 40+ concurrent requests just to shave
    time off a one-off preload isn't worth the risk of getting
    rate-limited)."""
    for symbol in MARKET_LIST:
        try:
            daily = fetcher.ensure_daily_reference(symbol)
            if daily:
                _bulk_load_state["results"][symbol] = daily
        except Exception:
            logger.exception("Bulk daily-load failed for %s", symbol)
        _bulk_load_state["done"] += 1
    _bulk_load_state["running"] = False


def _chart_cache_dir() -> Path:
    """The data/ directory both data_path() and chart_data_path() write
    into — resolved via tensor_store rather than hardcoded relative
    paths, so this stays correct if that ever moves."""
    return tensor_store.chart_data_path("_PLACEHOLDER_", "1d").parent

REGIME_LABELS = {"bullish": "Bullish", "bearish": "Bearish", "neutral": "Neutral"}
REGIME_COLORS = {"bullish": "#3f8f5f", "bearish": "#b3554f"}
# "full" = every regime timeframe agreed; "partial" = just enough to
# clear min_agreement. Same hues as REGIME_COLORS, lighter for partial.
REGIME_COLOR_TIERS = {
    "bullish": {"full": "#2f7a4f", "partial": "#7fb894"},
    "bearish": {"full": "#b3554f", "partial": "#d99490"},
}


def build_detail_panel(symbol: str, entry: dict, bot_status: dict) -> list:
    """Contents of the resizable detail panel for whichever symbol is
    currently selected. Price/change comes from the chart's own cache
    (market-data-cache); everything else — signal, regime, risk manager
    outcome, open position — comes read-only from gmx_bot's status.json.
    Nothing here can act on a trade; it only displays what the bot
    already decided.
    """
    if not symbol:
        return [html.Div("No symbol selected.", style={"color": "#888"})]

    bot_status = bot_status or {}
    signals = bot_status.get("signals", {})
    positions = bot_status.get("positions", {})
    is_stale = bot_status.get("is_stale", True)
    sig = signals.get(symbol)
    pos = positions.get(symbol)

    rows = [html.Div(symbol, style={"fontWeight": "700", "fontSize": "1.1em", "marginBottom": "4px"})]

    is_buyable = MARKET_BACKING.get(symbol)
    if is_buyable is not None:
        rows.append(html.Div([
            html.Span("Backing: ", style={"fontWeight": "600"}),
            html.Span(
                "Spot-backed (buyable)" if is_buyable else "Synthetic-only",
                style={"color": "#0a7d32" if is_buyable else "#b3702f"},
            ),
        ]))
        rows.append(html.Div(
            (f"At least one GMX pool genuinely holds {symbol} (or a token pegged to it)."
             if is_buyable else
             f"No real {symbol} backs this market on GMX \u2014 perp price exposure only, "
             f"not something you can actually buy/hold through it."),
            style={"fontSize": "0.85em", "color": "var(--muted)", "marginBottom": "4px"},
        ))
    rows.append(html.A(
        "Open in GMX \u2197", href="https://app.gmx.io/#/trade", target="_blank", rel="noopener noreferrer",
        style={"fontSize": "0.85em", "color": "#3f6fb3", "display": "inline-block", "marginBottom": "8px"},
    ))

    if entry:
        change = entry.get("change_pct")
        change_str, change_color = "", "inherit"
        if change is not None:
            sign = "+" if change >= 0 else ""
            change_str = f" ({sign}{change:.2f}%)"
            change_color = "#0a7d32" if change >= 0 else "#c92a2a"
        rows.append(html.Div([
            html.Span(format_price(entry["price"]), style={"fontWeight": "600"}),
            html.Span(change_str, style={"color": change_color, "marginLeft": "6px"}),
        ], style={"marginBottom": "8px"}))
    else:
        rows.append(html.Div(
            "No price data yet \u2014 click this symbol to load its chart.",
            style={"color": "#888", "fontSize": "0.9em", "marginBottom": "8px"},
        ))

    if is_stale:
        rows.append(html.Div(
            "Bot offline \u2014 details below are from its last run, not live.",
            style={"color": "#aaaaaa", "fontSize": "0.85em", "marginBottom": "6px"},
        ))

    if sig:
        regime = sig.get("regime")
        if regime:
            votes, total = sig.get("regime_votes"), sig.get("regime_total")
            vote_str = f" ({votes}/{total})" if votes is not None and total else ""
            rows.append(html.Div([
                html.Span("Regime: ", style={"fontWeight": "600"}),
                html.Span(REGIME_LABELS.get(regime, regime) + vote_str, style={"color": REGIME_COLORS.get(regime, "#666")}),
            ]))
            rows.append(html.Div(
                sig.get("regime_reason", ""),
                style={"fontSize": "0.88em", "color": "#666", "marginBottom": "6px"},
            ))

        direction = sig.get("direction", "flat")
        if direction in ("long", "short"):
            rows.append(html.Div([
                html.Span("Signal: ", style={"fontWeight": "600"}),
                html.Span(direction.upper(), style={"color": "#0a7d32" if direction == "long" else "#c92a2a"}),
            ]))
            rows.append(html.Div(
                sig.get("reason", ""), style={"fontSize": "0.88em", "color": "#666", "marginBottom": "6px"},
            ))

            if sig.get("risk_approved"):
                parts = []
                if sig.get("size_usd") is not None:
                    parts.append(f"size ${sig['size_usd']:.2f}")
                if sig.get("leverage") is not None:
                    parts.append(f"{sig['leverage']}x")
                if sig.get("stop_loss_pct") is not None:
                    parts.append(f"SL {sig['stop_loss_pct']}%")
                if sig.get("take_profit_pct") is not None:
                    parts.append(f"TP {sig['take_profit_pct']}%")
                rows.append(html.Div([
                    html.Span("Risk manager: ", style={"fontWeight": "600"}),
                    html.Span("approved" + (" \u2014 " + ", ".join(parts) if parts else ""), style={"color": "#0a7d32"}),
                ]))
            else:
                rows.append(html.Div([
                    html.Span("Risk manager: ", style={"fontWeight": "600"}),
                    html.Span("rejected", style={"color": "#c92a2a"}),
                ]))
                if sig.get("risk_reason"):
                    rows.append(html.Div(sig["risk_reason"], style={"fontSize": "0.88em", "color": "#666"}))
        else:
            rows.append(html.Div("No active entry signal this cycle.", style={"color": "#888", "fontSize": "0.9em"}))
    else:
        rows.append(html.Div("No bot data for this symbol yet.", style={"color": "#888", "fontSize": "0.9em"}))

    if pos:
        pnl = pos.get("unrealized_pnl_usd", 0.0)
        rows.append(html.Hr(style={"margin": "8px 0"}))
        rows.append(html.Div([
            html.Span("Open position: ", style={"fontWeight": "600"}),
            html.Span(pos.get("direction", "").upper()),
            html.Span(f" ${pos.get('notional_usd', 0):.2f} notional", style={"marginLeft": "6px", "color": "#666"}),
        ]))
        rows.append(html.Div(
            f"Unrealized PnL: ${pnl:.2f}",
            style={"color": "#0a7d32" if pnl >= 0 else "#c92a2a", "fontSize": "0.9em"},
        ))

    return rows

def maintenance_button_style() -> dict:
    """Shared look for the Chart settings maintenance buttons (clear
    cache / bulk-load daily) — distinct from period_button_style since
    these aren't a selectable set, just one-off actions."""
    return {
        "fontSize": "0.8em", "padding": "5px 8px", "cursor": "pointer",
        "backgroundColor": "var(--panel-bg)", "color": "var(--text)",
        "border": "1px solid var(--border)", "borderRadius": "4px", "textAlign": "left",
    }


def period_button_style(is_selected: bool) -> dict:
    """TradingView-style timeframe pill — selected one filled, others
    outlined. Shared between the initial layout render and
    update_period_button_styles so both stay in sync by construction."""
    return {
        "padding": "4px 10px", "fontSize": "0.8em", "cursor": "pointer",
        "border": "1px solid var(--border)", "borderRadius": "4px",
        "backgroundColor": "var(--selected-bg)" if is_selected else "var(--panel-bg)",
        "color": "var(--text)", "fontWeight": "700" if is_selected else "400",
    }


app = dash.Dash(__name__)
app.title = "TensorLang GMX Charts"

# Theme via CSS variables + a single class toggle on the root div (see
# "app-root" below and toggle_theme()) — deliberately NOT done by having
# a callback rewrite every component's `style` dict on theme change.
# detail-panel is user-resizable via plain CSS (resize: vertical), which
# sets an explicit height directly in the DOM's inline style the moment
# someone drags it; a callback that re-outputs a whole new style dict
# for that element would clobber that height back to the default on
# every theme toggle. Components below reference var(--...) inside their
# *initial* inline style, which never gets rewritten — only the CSS
# variables' resolved values change when .theme-dark is added/removed.
THEME_CSS = """
:root {
    --bg: #ffffff; --text: #1a1a1a; --muted: #666666; --border: #cccccc;
    --panel-bg: #ffffff; --selected-bg: #e8f0fe; --input-bg: #ffffff;
    --synthetic-bg: #fdf3e0;
}
body.theme-dark {
    --bg: #121212; --text: #e2e2e2; --muted: #9a9a9a; --border: #3a3a3a;
    --panel-bg: #1a1a1a; --selected-bg: #2a3a52; --input-bg: #242424;
    --synthetic-bg: #362c1a;
}
body { background: var(--bg); color: var(--text); margin: 0; }
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{THEME_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    id="app-root",
    style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif",
           "backgroundColor": "var(--bg)", "color": "var(--text)"},
    children=[
        html.Div(
            style={"flex": "1", "padding": "10px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"},
                    children=[
                        html.Details(
                            style={"marginBottom": "10px"},
                            children=[
                                html.Summary("Chart settings", style={"cursor": "pointer", "fontWeight": "600"}),
                                dcc.Checklist(
                                    id="chart-options",
                                    options=CHART_OPTIONS,
                                    value=DEFAULT_CHART_OPTIONS,
                                    style={"marginTop": "6px"},
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                                html.Hr(style={"margin": "8px 0", "borderColor": "var(--border)"}),
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px", "maxWidth": "220px"},
                                    children=[
                                        html.Button(
                                            "\U0001F5D1 Clear chart cache", id="clear-cache-btn", n_clicks=0,
                                            style=maintenance_button_style(),
                                        ),
                                        html.Div(id="clear-cache-status", style={"fontSize": "0.8em", "color": "var(--muted)"}),
                                        html.Button(
                                            "\u2B07 Load all daily data", id="load-all-daily-btn", n_clicks=0,
                                            style=maintenance_button_style(),
                                        ),
                                        html.Div(id="bulk-load-status", style={"fontSize": "0.8em", "color": "var(--muted)"}),
                                    ],
                                ),
                                dcc.ConfirmDialog(
                                    id="clear-cache-confirm",
                                    message=(
                                        "Delete every cached chart file in gmx_charts/data/? "
                                        "This can't be undone from here \u2014 each symbol refetches "
                                        "from GMX the next time it's viewed (or via 'Load all daily data')."
                                    ),
                                ),
                                dcc.Interval(id="bulk-load-poll-interval", interval=1000, n_intervals=0, disabled=True),
                            ],
                        ),
                        html.Button(
                            id="theme-toggle-btn", n_clicks=0,
                            style={
                                "fontSize": "0.8em", "padding": "4px 10px", "cursor": "pointer",
                                "backgroundColor": "var(--panel-bg)", "color": "var(--text)",
                                "border": "1px solid var(--border)", "borderRadius": "4px",
                            },
                        ),
                    ],
                ),
                # TradingView-style timeframe row instead of a dropdown —
                # selected-period is the source of truth (a dcc.Store,
                # not a form control's own value), so both the click
                # handler and the restyle-on-selection callback below can
                # read/write it independently.
                dcc.Store(id="selected-period", data=DEFAULT_PERIOD),
                html.Div(
                    id="period-row",
                    style={"display": "flex", "gap": "4px", "marginBottom": "10px", "flexWrap": "wrap"},
                    children=[
                        html.Button(
                            p, id={"type": "period-btn", "index": p}, n_clicks=0,
                            style=period_button_style(p == DEFAULT_PERIOD),
                        )
                        for p in PERIODS
                    ],
                ),
                dcc.Graph(id="candlestick-chart", style={"height": "85vh"}),
                # Keeps the currently-open chart (and its sidebar price
                # row) current without needing a click — get_bars() now
                # refreshes its own cache once stale (see
                # fetcher.ensure_chart_history_fresh), this just gives it
                # a reason to re-check periodically instead of only on
                # market/period/theme change. Deliberately only refreshes
                # the SELECTED symbol, not all 40+ sidebar rows, so this
                # doesn't multiply API calls for symbols nobody's looking
                # at right now — those still refresh on click, which now
                # actually works (that was the bug).
                dcc.Interval(id="chart-refresh-interval", interval=30000, n_intervals=0),
            ],
        ),
        html.Div(
            style={
                "width": "260px", "borderLeft": "1px solid var(--border)",
                "padding": "10px", "display": "flex", "flexDirection": "column",
                "height": "85vh",  # matches the chart's height so the sidebar doesn't grow the page
            },
            children=[
                # This header sits OUTSIDE the scrolling div below, so it
                # never scrolls out of view — that's the whole fix, no
                # position:sticky needed since it's just not part of the
                # scrollable area in the first place.
                html.Div(
                    style={
                        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                        "flexShrink": 0, "paddingBottom": "8px",
                    },
                    children=[
                        html.H4("Markets", style={"margin": 0}),
                        html.Span(id="bot-status-label", style={"fontSize": "0.75em"}),
                        html.Button(
                            "Next signal \u2192", id="next-signal-btn", n_clicks=0,
                            style={"fontSize": "0.75em", "padding": "4px 8px", "cursor": "pointer"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "fontSize": "0.7em", "color": "var(--muted)", "paddingBottom": "6px",
                        "flexShrink": 0, "display": "flex", "gap": "12px", "whiteSpace": "nowrap",
                        "flexWrap": "wrap",
                    },
                    children=[
                        html.Span([
                            html.Span("\u25cf", style={"fontSize": "1.2em", "verticalAlign": "middle"}),
                            " approved",
                        ]),
                        html.Span([
                            html.Span("\u25cb", style={"fontSize": "1.2em", "verticalAlign": "middle"}),
                            " rejected",
                        ]),
                        html.Span([
                            html.Span(style={
                                "display": "inline-block", "width": "10px", "height": "10px",
                                "backgroundColor": "var(--synthetic-bg)", "border": "1px solid var(--border)",
                                "verticalAlign": "middle", "marginRight": "3px",
                            }),
                            "synthetic-only",
                        ]),
                    ],
                ),
                dcc.Checklist(
                    id="buyable-only-filter",
                    options=[{"label": " Show buyable (spot-backed) only", "value": "buyable_only"}],
                    value=[],
                    style={"fontSize": "0.75em", "color": "var(--muted)", "paddingBottom": "6px", "flexShrink": 0},
                ),
                html.Div(
                    id="market-list-scroll",
                    style={"overflowY": "auto", "flex": "1 1 auto", "minHeight": "0"},
                    children=[
                        html.Ul(
                            id="market-list",
                            style={"listStyle": "none", "padding": 0, "margin": 0},
                            # These <Li>s are created ONCE, here, and never
                            # regenerated — see update_market_rows below, which
                            # patches each row's `children` in place instead of
                            # replacing the <Li>s themselves. Recreating them
                            # on every price update was resetting n_clicks to 0
                            # on components Dash's pattern-matching click
                            # tracker depends on, which is the most likely
                            # cause of "changing the timeframe forgets the
                            # selected symbol" — the click-tracker misfiring
                            # off of components that were never actually
                            # clicked, right after a fresh regeneration.
                            children=[
                                html.Li(
                                    market_row(symbol, None, False),
                                    id={"type": "market-item", "index": symbol},
                                    n_clicks=0,
                                    style={"cursor": "pointer"},
                                    **{"data-symbol": symbol},  # plain attribute for the JS scroll-into-view below, unrelated to Dash's click id
                                )
                                for symbol in MARKET_LIST
                            ],
                        ),
                    ],
                ),
                # Resizable via plain CSS (resize: vertical) — no JS
                # needed, drag the bottom-right corner. Not supported on
                # Safari desktop for non-textarea elements as of this
                # writing; it just won't be draggable there, the fixed
                # default height still works fine.
                html.Div(
                    id="detail-panel",
                    style={
                        "borderTop": "1px solid var(--border)", "marginTop": "6px", "paddingTop": "8px",
                        "flexShrink": 0, "height": "200px", "minHeight": "90px", "maxHeight": "65vh",
                        "overflow": "auto", "resize": "vertical", "fontSize": "1em",
                    },
                    children=build_detail_panel(MARKET_LIST[0] if MARKET_LIST else None, None, {}),
                ),
            ],
        ),
        html.Div(id="scroll-sync-dummy", style={"display": "none"}),
        dcc.Store(id="selected-market", data=MARKET_LIST[0] if MARKET_LIST else "ETH"),
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        dcc.Store(id="market-data-cache", data={}),  # {symbol: {"price": ..., "change_pct": ...}}
        dcc.Interval(id="bot-status-interval", interval=BOT_STATUS_POLL_MS, n_intervals=0),
        dcc.Store(id="bot-status-cache", data={}),  # raw contents of gmx_bot's status.json
    ],
)


@app.callback(
    Output("bot-status-cache", "data"),
    Input("bot-status-interval", "n_intervals"),
)
def poll_bot_status(_n):
    status = load_bot_status()
    generated_at = status.get("generated_at") if status else None
    status["is_stale"] = generated_at is None or (time.time() - generated_at) > BOT_STALE_AFTER_SECONDS
    return status


# Scrolls the selected row into view inside the sidebar's own scroll
# container (#market-list-scroll), not the whole page. Runs in the
# browser rather than as a Python callback since "scroll this element
# into view" isn't a property Dash tracks server-side — there's no
# component prop to set that would cause it. The dummy output is just
# a place to put a required Output; nothing reads scroll-sync-dummy's
# children back.
app.clientside_callback(
    """
    function(selected_market) {
        if (!selected_market) { return window.dash_clientside.no_update; }
        setTimeout(function() {
            var el = document.querySelector('[data-symbol="' + selected_market + '"]');
            if (el) { el.scrollIntoView({block: "nearest", behavior: "smooth"}); }
        }, 50);
        return window.dash_clientside.no_update;
    }
    """,
    Output("scroll-sync-dummy", "children"),
    Input("selected-market", "data"),
)


@app.callback(
    Output("selected-market", "data", allow_duplicate=True),
    Input({"type": "market-item", "index": dash.ALL}, "n_clicks"),
    State({"type": "market-item", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def on_market_click(_n_clicks_list, _ids):
    triggered = dash.callback_context.triggered_id
    if triggered is None:
        return dash.no_update
    return triggered["index"]


@app.callback(
    Output({"type": "market-item", "index": dash.ALL}, "style"),
    Input("buyable-only-filter", "value"),
    State({"type": "market-item", "index": dash.ALL}, "id"),
)
def filter_market_rows(filter_value, ids):
    """Toggles display:none on rows rather than removing/recreating the
    <Li> elements — recreating them (instead of patching their
    `children` in place, as update_market_rows already does) was the
    exact bug behind an earlier issue where changing a filter/timeframe
    reset Dash's pattern-matching click tracker on components that were
    never actually clicked. Only explicitly-synthetic symbols (False)
    get hidden; unknown backing (missing from MARKET_BACKING) stays
    visible rather than being hidden speculatively."""
    buyable_only = "buyable_only" in (filter_value or [])
    styles = []
    for id_dict in ids:
        hidden = buyable_only and MARKET_BACKING.get(id_dict["index"]) is False
        styles.append({"cursor": "pointer", "display": "none" if hidden else "block"})
    return styles


@app.callback(
    Output("selected-market", "data", allow_duplicate=True),
    Input("next-signal-btn", "n_clicks"),
    State("bot-status-cache", "data"),
    State("selected-market", "data"),
    prevent_initial_call=True,
)
def jump_to_next_signal(_n_clicks, bot_status, current_market):
    """Cycles through symbols that currently have a non-flat bot signal,
    in the same order they appear in the sidebar. Wraps around. No-ops
    (dash.no_update) if nothing has a signal right now."""
    bot_status = bot_status or {}
    signals = bot_status.get("signals", {})
    signal_symbols = [s for s in MARKET_LIST if signals.get(s, {}).get("direction") in ("long", "short")]
    if not signal_symbols:
        return dash.no_update
    if current_market in signal_symbols:
        next_idx = (signal_symbols.index(current_market) + 1) % len(signal_symbols)
    else:
        next_idx = 0
    return signal_symbols[next_idx]


@app.callback(
    Output("candlestick-chart", "figure"),
    Output("market-data-cache", "data"),
    Input("selected-market", "data"),
    Input("selected-period", "data"),
    Input("chart-options", "value"),
    Input("bot-status-cache", "data"),
    Input("theme-store", "data"),
    Input("chart-refresh-interval", "n_intervals"),
    State("market-data-cache", "data"),
)
def update_chart(market, period, chart_options, bot_status, theme, _n_intervals, cache):
    cache = dict(cache or {})
    if not market:
        return go.Figure(), cache
    bot_status = bot_status or {}
    entry_price = bot_status.get("positions", {}).get(market, {}).get("entry_price")
    fig, entry = build_figure_and_data(
        market, period, set(chart_options or []), entry_price=entry_price, theme=theme or "light",
    )
    if entry is not None:
        cache[market] = entry
    return fig, cache


@app.callback(
    Output("clear-cache-confirm", "displayed"),
    Input("clear-cache-btn", "n_clicks"),
    prevent_initial_call=True,
)
def ask_clear_cache(_n_clicks):
    return True


@app.callback(
    Output("market-data-cache", "data", allow_duplicate=True),
    Output("clear-cache-status", "children"),
    Output("candlestick-chart", "figure", allow_duplicate=True),
    Input("clear-cache-confirm", "submit_n_clicks"),
    State("selected-market", "data"),
    State("selected-period", "data"),
    State("chart-options", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def clear_cache(_submit_n_clicks, market, period, chart_options, theme):
    """Deletes every cached .npy in gmx_charts/data/ (both the chart
    cache and the fixed-window indicator files live there) and
    immediately refetches+redraws whatever's currently selected, so the
    person doesn't have to reselect it themselves to see the effect.
    Every OTHER sidebar row goes back to blank until it's clicked again
    or 'Load all daily data' is used — same on-demand-backfill design
    as the rest of this app, just reset to its starting state."""
    removed = 0
    for f in _chart_cache_dir().glob("*.npy"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            logger.exception("Failed to remove cache file %s", f)

    fig = go.Figure()
    new_cache = {}
    if market:
        fig, entry = build_figure_and_data(market, period, set(chart_options or []), theme=theme or "light")
        if entry is not None:
            new_cache[market] = entry

    return new_cache, f"Cleared {removed} cached file(s).", fig


@app.callback(
    Output("bulk-load-poll-interval", "disabled"),
    Output("bulk-load-status", "children"),
    Input("load-all-daily-btn", "n_clicks"),
    prevent_initial_call=True,
)
def start_bulk_load(_n_clicks):
    if _bulk_load_state["running"]:
        return False, dash.no_update  # already running — just make sure the poller stays on
    _bulk_load_state.update({"running": True, "done": 0, "total": len(MARKET_LIST), "results": {}})
    threading.Thread(target=_bulk_load_daily_worker, daemon=True).start()
    return False, f"Loading daily data\u2026 0/{len(MARKET_LIST)}"


@app.callback(
    Output("bulk-load-poll-interval", "disabled", allow_duplicate=True),
    Output("bulk-load-status", "children", allow_duplicate=True),
    Output("market-data-cache", "data", allow_duplicate=True),
    Input("bulk-load-poll-interval", "n_intervals"),
    State("market-data-cache", "data"),
    prevent_initial_call=True,
)
def poll_bulk_load(_n_intervals, cache):
    cache = dict(cache or {})
    cache.update(_bulk_load_state["results"])
    done, total = _bulk_load_state["done"], _bulk_load_state["total"]
    running = _bulk_load_state["running"]
    status = f"Loading daily data\u2026 {done}/{total}" if running else f"Loaded daily data for {done}/{total} symbol(s)."
    return (not running), status, cache


@app.callback(
    Output({"type": "period-btn", "index": dash.ALL}, "style"),
    Input("selected-period", "data"),
    State({"type": "period-btn", "index": dash.ALL}, "id"),
)
def update_period_button_styles(selected_period, ids):
    return [period_button_style(id_dict["index"] == selected_period) for id_dict in ids]


@app.callback(
    Output("selected-period", "data"),
    Input({"type": "period-btn", "index": dash.ALL}, "n_clicks"),
    State({"type": "period-btn", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def on_period_click(_n_clicks_list, _ids):
    triggered = dash.callback_context.triggered_id
    if triggered is None:
        return dash.no_update
    return triggered["index"]


@app.callback(
    Output("app-root", "className"),
    Output("theme-toggle-btn", "children"),
    Input("theme-store", "data"),
)
def apply_theme(theme):
    theme = theme or "light"
    label = "\u2600\ufe0f Light mode" if theme == "dark" else "\U0001F319 Dark mode"
    return ("theme-dark" if theme == "dark" else ""), label


@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(_n_clicks, current_theme):
    return "dark" if (current_theme or "light") == "light" else "light"


@app.callback(
    Output({"type": "market-item", "index": dash.ALL}, "children"),
    Input("market-data-cache", "data"),
    Input("selected-market", "data"),
    Input("bot-status-cache", "data"),
    State({"type": "market-item", "index": dash.ALL}, "id"),
)
def update_market_rows(cache, selected_market, bot_status, ids):
    cache = cache or {}
    bot_status = bot_status or {}
    signals = bot_status.get("signals", {})
    positions = bot_status.get("positions", {})
    is_stale = bot_status.get("is_stale", True)  # no data at all counts as stale/offline
    return [
        market_row(
            id_dict["index"],
            cache.get(id_dict["index"]),
            id_dict["index"] == selected_market,
            bot_signal=signals.get(id_dict["index"]),
            has_position=id_dict["index"] in positions,
            bot_stale=is_stale,
        )
        for id_dict in ids
    ]


@app.callback(
    Output("bot-status-label", "children"),
    Output("bot-status-label", "style"),
    Input("bot-status-cache", "data"),
)
def update_bot_status_label(bot_status):
    bot_status = bot_status or {}
    base_style = {"fontSize": "0.75em"}
    if bot_status.get("is_stale", True):
        return "\u25cf Bot offline", {**base_style, "color": "#aaaaaa"}
    equity = bot_status.get("equity_usd")
    equity_str = f" (${equity:,.2f})" if equity is not None else ""
    return f"\u25cf Bot live{equity_str}", {**base_style, "color": "#0a7d32"}


@app.callback(
    Output("detail-panel", "children"),
    Input("selected-market", "data"),
    Input("market-data-cache", "data"),
    Input("bot-status-cache", "data"),
)
def update_detail_panel(selected_market, cache, bot_status):
    cache = cache or {}
    return build_detail_panel(selected_market, cache.get(selected_market), bot_status)


if __name__ == "__main__":
    app.run(debug=True)
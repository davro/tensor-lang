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
import sys
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

# 1w/1mo aren't in this list because they're not sent to GMX directly —
# see RESAMPLE_RULES below. GMX's own accepted period strings for its
# oracle-keeper candles endpoint aren't documented anywhere confirmable
# from the sandbox this was built in, so these two are built locally
# instead of gambling on an unverified API value.
PERIODS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"]
DEFAULT_PERIOD = "1h"
FALLBACK_MARKETS = ["ETH", "BTC"]  # used only if fetch_markets() fails
CHART_CANDLES = 5000  # ~208 days at 1h — see fetcher.GMX_CANDLE_LIMIT for the hard ceiling

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


def add_overlays(fig: go.Figure, bars: np.ndarray, options: set) -> None:
    """Adds the optional current-price line, plus whichever overlay
    indicators from chart_indicators.OVERLAY_INDICATORS are toggled on.
    Adding a new indicator to that registry needs no change here —
    this just iterates it generically.
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
    the normal GMX-backed chart cache.
    """
    if period in RESAMPLE_RULES:
        base_period, rule = RESAMPLE_RULES[period]
        base_bars = fetcher.ensure_chart_history(market, base_period, n_candles=CHART_CANDLES)
        return resample_bars(base_bars, rule)
    return fetcher.ensure_chart_history(market, period, n_candles=CHART_CANDLES)


def build_figure_and_data(market: str, period: str, options: set = frozenset()):
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
    try:
        bars = get_bars(market, period)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Couldn't load {market} {period}: {e}")
        return fig, None

    if len(bars) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{market} {period}: no data yet")
        return fig, None

    fig = go.Figure(data=[go.Candlestick(
        x=pd.to_datetime(bars[:, tensor_store.COL_TIMESTAMP], unit="s"),
        open=bars[:, tensor_store.COL_OPEN],
        high=bars[:, tensor_store.COL_HIGH],
        low=bars[:, tensor_store.COL_LOW],
        close=bars[:, tensor_store.COL_CLOSE],
        name=market,
    )])
    add_overlays(fig, bars, options)
    fig.update_layout(
        title=f"{market} / USD — {period}",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=20),
    )
    fig.update_yaxes(tickformat=y_axis_tickformat(bars))

    daily = fetcher.ensure_daily_reference(market)
    data = {
        "price": float(bars[-1, tensor_store.COL_CLOSE]),
        "change_pct": daily["change_pct"] if daily else None,
        "period": period,
    }
    return fig, data


def market_row(symbol: str, entry: dict, is_selected: bool) -> html.Div:
    """One sidebar row: symbol (left, flexible), price (right-aligned,
    fixed width), daily % change (right-aligned, fixed width, color-
    coded). Fixed-width columns are what actually gives the table-like
    alignment regardless of symbol length — text-only padding can't do
    that once symbol lengths vary (0G vs APE_deprecated). The selected
    row also gets a small badge showing which chart period its PRICE
    reflects — the % change itself always means "vs yesterday", so the
    badge is only about the price's freshness, not the % figure.
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

    symbol_children = [symbol]
    if period_badge:
        symbol_children.append(html.Span(
            period_badge,
            style={
                "marginLeft": "6px", "fontSize": "0.7em", "color": "#555",
                "backgroundColor": "#dde3ea", "borderRadius": "3px", "padding": "1px 4px",
            },
        ))

    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "width": "100%",
            "backgroundColor": "#e8f0fe" if is_selected else "transparent",
            "fontWeight": "600" if is_selected else "normal",
            "padding": "2px 4px",
            "borderRadius": "4px",
        },
        children=[
            html.Span(symbol_children, style={"flex": "1 1 auto", "overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Span(price_str, style={"flex": "0 0 90px", "textAlign": "right"}),
            html.Span(change_str, style={"flex": "0 0 60px", "textAlign": "right", "color": change_color}),
        ],
    )


MARKET_LIST = get_market_list()

app = dash.Dash(__name__)
app.title = "TensorLang GMX Charts"

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    children=[
        html.Div(
            style={"flex": "1", "padding": "10px"},
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
                    ],
                ),
                dcc.Dropdown(
                    id="period-dropdown",
                    options=[{"label": p, "value": p} for p in PERIODS],
                    value=DEFAULT_PERIOD,
                    clearable=False,
                    style={"width": "150px", "marginBottom": "10px"},
                ),
                dcc.Graph(id="candlestick-chart", style={"height": "85vh"}),
            ],
        ),
        html.Div(
            style={
                "width": "260px", "borderLeft": "1px solid #ccc",
                "overflowY": "auto", "padding": "10px",
            },
            children=[
                html.H4("Markets"),
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
                        )
                        for symbol in MARKET_LIST
                    ],
                ),
            ],
        ),
        dcc.Store(id="selected-market", data=MARKET_LIST[0] if MARKET_LIST else "ETH"),
        dcc.Store(id="market-data-cache", data={}),  # {symbol: {"price": ..., "change_pct": ...}}
    ],
)


@app.callback(
    Output("selected-market", "data"),
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
    Output("candlestick-chart", "figure"),
    Output("market-data-cache", "data"),
    Input("selected-market", "data"),
    Input("period-dropdown", "value"),
    Input("chart-options", "value"),
    State("market-data-cache", "data"),
)
def update_chart(market, period, chart_options, cache):
    cache = dict(cache or {})
    if not market:
        return go.Figure(), cache
    fig, entry = build_figure_and_data(market, period, set(chart_options or []))
    if entry is not None:
        cache[market] = entry
    return fig, cache


@app.callback(
    Output({"type": "market-item", "index": dash.ALL}, "children"),
    Input("market-data-cache", "data"),
    Input("selected-market", "data"),
    State({"type": "market-item", "index": dash.ALL}, "id"),
)
def update_market_rows(cache, selected_market, ids):
    cache = cache or {}
    return [
        market_row(id_dict["index"], cache.get(id_dict["index"]), id_dict["index"] == selected_market)
        for id_dict in ids
    ]


if __name__ == "__main__":
    app.run(debug=True)

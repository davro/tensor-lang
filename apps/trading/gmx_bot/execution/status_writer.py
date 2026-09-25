"""
Writes a small JSON snapshot of the bot's live state (open positions +
per-symbol signals) after every cycle, purely for gmx_charts to read.

One-way and read-only from the bot's side: this module makes no chain
calls, and gmx_charts never imports anything from execution/ or risk/ —
it only ever reads this file off disk. Keeping that boundary one-way
means a bug in the chart's Dash/Plotly code can never reach the wallet
path.

Written atomically (temp file + os.replace) so a poller reading this
file on a timer never sees a half-written JSON document mid-write.
"""
import json
import os
import tempfile
import time
from typing import Dict, Iterable, List, Tuple

from risk.risk_manager import OpenPosition, RiskDecision
from strategy.regime import RegimeReading
from strategy.signal import MultiTimeframeSignal

STATUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "status.json"
)


def to_chart_symbol(bot_symbol: str) -> str:
    """'BTC/USD' -> 'BTC', matching gmx_charts' bare-ticker file names
    (data/{market}_{period}_chart.npy)."""
    return bot_symbol.split("/")[0].upper()


def write_status(
    path: str,
    open_positions: List[OpenPosition],
    signal_results: Iterable[Tuple[str, MultiTimeframeSignal, RegimeReading]],
    decisions: Dict[str, RiskDecision],
    account_equity_usd: float,
) -> None:
    """Called once per cycle from main.run_cycle(), after everything for
    that cycle has already been computed — this only serializes it.

    signal_results carries the regime reading alongside each filtered
    signal (not just folded into its reason string) so gmx_charts can
    render regime as its own field — a colored symbol name, a badge,
    whatever it wants — rather than parsing a sentence. decisions is
    keyed by the bot's own "BASE/USD" symbol (not the chart's bare
    ticker) since that's what risk_manager.evaluate() was called with.
    """
    payload = {
        "generated_at": time.time(),
        "equity_usd": account_equity_usd,
        "positions": {
            to_chart_symbol(p.symbol): {
                "direction": p.direction.value,
                "notional_usd": p.notional_usd,
                "unrealized_pnl_usd": p.unrealized_pnl_usd,
                # None today — OpenPosition doesn't expose entry price yet.
                # Wired up defensively (getattr) so the moment that field
                # is added on the bot side, it starts flowing here and to
                # the chart's entry-price line with no further changes.
                "entry_price": getattr(p, "entry_price", None),
            }
            for p in open_positions
        },
        "signals": {},
    }

    for symbol, sig, regime in signal_results:
        decision = decisions.get(symbol)
        entry = {
            "direction": sig.direction.value,
            "reason": sig.reason,
            "entry_rsis": sig.entry_rsis,
            "trend_rsis": sig.trend_rsis,
            "regime": regime.regime.value,
            "regime_reason": regime.reason,
            "regime_rsis": regime.rsis,
            "regime_votes": regime.votes,
            "regime_total": regime.total,
            "risk_approved": decision.approved if decision else False,
            "risk_reason": decision.reason if decision else None,
        }
        # Sizing fields only mean something for an approved decision —
        # a rejected one may not even set them, so this stays defensive
        # (getattr) rather than assuming every RiskDecision has all four.
        if decision and decision.approved:
            entry["size_usd"] = getattr(decision, "size_usd", None)
            entry["leverage"] = getattr(decision, "leverage", None)
            entry["stop_loss_pct"] = getattr(decision, "stop_loss_pct", None)
            entry["take_profit_pct"] = getattr(decision, "take_profit_pct", None)
        payload["signals"][to_chart_symbol(symbol)] = entry

    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".status_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
"""
TensorLang trading bot — main loop.

Flow each cycle:
  1. Fetch real open positions + account equity from GMX (execution/positions.py).
  2. For each open position, check the RSI-reversion exit on the
     configured exit_timeframe. If triggered, close it.
  3. For symbols with NO open position, generate ONE combined signal per
     symbol: RSI across signal_timeframes (5m/15m/1h) must reach a vote
     threshold (min_entry_agreement) before a direction is even
     considered, then trend_timeframes (4h/1d) act as a veto/scale
     filter. This replaced firing an independent signal per timeframe,
     which could otherwise send 2-3 separate (sometimes conflicting)
     order attempts for the same symbol within one cycle.

Both order placement and closes go through execution/gmx_client.py,
which defaults to dry-run (execution.debug_mode = True) — nothing is
submitted on-chain until that's explicitly turned off.

Run with: python main.py
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env before settings reads it

from config.settings import settings
from data.market_data import CachingDataProvider, GmxRestDataProvider
from execution.gmx_client import GmxExecutionClient
from execution.positions import PositionTracker
from execution.status_writer import STATUS_PATH, write_status
from risk.risk_manager import RiskManager
from strategy.regime import apply_regime_filter, classify_regime
from strategy.signal import Direction, MultiTimeframeSignal, generate_exit_signal, generate_multi_timeframe_signal

# LOG_LEVEL=DEBUG python3 main.py to see every symbol's RSI each cycle
# (signal.display() below), not just the ones that crossed a threshold —
# useful for confirming the pipeline is still alive during a quiet
# market rather than wondering if something broke.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")
# eth_defi's own position-lookup logs its REST/GraphQL fallback attempts at
# INFO on every call, which gets noisy once you're scanning many symbols.
logging.getLogger("eth_defi.gmx.core.open_positions").setLevel(logging.WARNING)


def build_data_provider() -> CachingDataProvider:
    base = GmxRestDataProvider(chain=settings.chain.chain)
    return CachingDataProvider(base, ttl_seconds=settings.execution.poll_interval_seconds)


def classify_position_regimes(data_provider, open_positions) -> dict:
    """Regime for each currently-held symbol, computed ONCE per cycle
    here and reused both by handle_exits (regime-aware exit widening/
    tightening) and handle_entries (so the chart's sidebar tint and
    detail panel show regime for held positions too, not just
    candidates) — avoids fetching the same weekly/monthly candles for
    the same symbol twice in the same cycle."""
    if not open_positions:
        return {}

    def fetch(pos):
        return pos.symbol, classify_regime(
            symbol=pos.symbol, data=data_provider, rsi_cfg=settings.rsi,
            regime_timeframes=getattr(settings.market, "regime_timeframes", ["1d", "1w", "1M"]),
        )

    regimes = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch, pos): pos.symbol for pos in open_positions}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, regime = future.result()
                regimes[sym] = regime
            except Exception:
                logger.exception("Failed to classify regime for open position %s", symbol)
    return regimes


def handle_exits(data_provider, position_tracker: PositionTracker, execution_client: GmxExecutionClient,
                  open_positions, position_regimes: dict = None):
    """Check every open position for an RSI-reversion exit and close it
    if triggered. Returns the set of symbols that got closed this cycle."""
    position_regimes = position_regimes or {}
    closed_symbols = set()
    for pos in open_positions:
        try:
            regime = position_regimes.get(pos.symbol)
            exit_signal = generate_exit_signal(
                symbol=pos.symbol,
                timeframe=settings.market.exit_timeframe,
                position_direction=pos.direction,
                data=data_provider,
                rsi_cfg=settings.rsi,
                regime_bias=regime.regime.value if regime else None,
            )
        except Exception:
            logger.exception("Failed to generate exit signal for %s", pos.symbol)
            continue

        logger.debug("EXIT CHECK %s: %s", pos.symbol, exit_signal.reason)
        if not exit_signal.should_exit:
            continue

        logger.info("EXIT %s: %s", pos.symbol, exit_signal.reason)
        try:
            result = execution_client.close_position(pos, exit_signal.reason)
            logger.info("Close result: %s", result)
            if result.submitted or result.debug_only:
                closed_symbols.add(pos.symbol)
        except Exception:
            logger.exception("Failed to close position for %s", pos.symbol)

    return closed_symbols


def handle_entries(data_provider, risk_manager: RiskManager, execution_client: GmxExecutionClient,
                    open_positions, account_equity_usd: float, skip_symbols: set,
                    position_regimes: dict = None):
    """One combined signal per symbol (RSI voted across signal_timeframes,
    filtered by trend_timeframes) instead of one independent signal per
    (symbol, timeframe) pair — avoids the same symbol getting multiple,
    sometimes conflicting, order attempts within a single cycle."""
    position_regimes = position_regimes or {}
    signal_count = 0
    error_count = 0
    decisions = {}  # {symbol: RiskDecision} — kept for status_writer, not just used inline below

    open_symbols = {p.symbol for p in open_positions} | skip_symbols
    candidate_symbols = [s for s in settings.market.symbols if s not in open_symbols]

    def fetch_signal(symbol):
        signal = generate_multi_timeframe_signal(
            symbol=symbol, data=data_provider, rsi_cfg=settings.rsi,
            entry_timeframes=settings.market.signal_timeframes,
            trend_timeframes=settings.market.trend_timeframes,
            min_agreement=settings.market.min_entry_agreement,
        )
        # Regime is a filter, not a signal generator (see strategy/regime.py):
        # it never turns FLAT into a trade, it only scales/vetoes a trade
        # that fights the medium/long-term bias. Computed even for FLAT
        # signals so status.json can show regime context regardless, and
        # returned alongside the filtered signal (not just folded into its
        # reason string) so the chart can render it as its own field.
        regime = classify_regime(
            symbol=symbol, data=data_provider, rsi_cfg=settings.rsi,
            regime_timeframes=getattr(settings.market, "regime_timeframes", ["1d", "1w", "1M"]),
        )
        return symbol, apply_regime_filter(signal, regime), regime

    # Network-bound (candle fetches across 3 entry + 2 trend timeframes
    # per symbol), so a thread pool matters once you're scanning 40+ symbols.
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_signal, symbol): symbol for symbol in candidate_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception:
                error_count += 1
                logger.exception("Failed to generate signal for %s", symbol)

    # Held positions get no NEW entry signal (handle_exits owns the exit
    # decision for these), but they still get a regime entry appended
    # here — reusing what classify_position_regimes already computed
    # this cycle — purely so status.json (and the chart) shows regime
    # tint/detail for a symbol even while it's in a trade, instead of
    # going blank the moment a position opens.
    for pos in open_positions:
        regime = position_regimes.get(pos.symbol)
        if regime is not None:
            held_signal = MultiTimeframeSignal(pos.symbol, Direction.FLAT, 0.0, "position already open", {}, {})
            results.append((pos.symbol, held_signal, regime))

    # Risk evaluation touches shared state (exposure totals), so keep it
    # sequential even though fetching was parallel.
    for symbol, signal, regime in sorted(results, key=lambda r: r[0]):
        logger.debug(signal.display())
        if signal.direction.value != "flat":
            signal_count += 1
            logger.info("SIGNAL %s", signal.display())

        decision = risk_manager.evaluate(signal, open_positions, account_equity_usd)
        decisions[symbol] = decision
        if not decision.approved:
            if signal.direction.value != "flat":
                logger.info("Risk manager rejected %s: %s", symbol, decision.reason)
            continue

        result = execution_client.place_order(symbol, signal.direction, decision)
        logger.info("Order result: %s", result)

    if signal_count == 0:
        _log_near_misses(results)

    return signal_count, error_count, results, decisions


def _log_near_misses(results, top_n: int = 3) -> None:
    """Called only when a cycle produces zero entry signals. Ranks flat
    signals by how close their nearest entry-timeframe RSI came to the
    oversold/overbought line and logs the top few, so 'no signals' reads
    as 'nothing crossed the threshold right now' instead of looking
    identical to the pipeline having silently stopped working."""
    def gap_to_threshold(signal) -> float:
        gaps = [
            min(abs(rsi - settings.rsi.oversold), abs(rsi - settings.rsi.overbought))
            for rsi in signal.entry_rsis.values() if rsi is not None
        ]
        return min(gaps) if gaps else float("inf")

    flat_results = [(symbol, sig) for symbol, sig, _regime in results if sig.direction.value == "flat"]
    nearest = sorted(flat_results, key=lambda r: gap_to_threshold(r[1]))[:top_n]
    if nearest:
        summary = "; ".join(f"{symbol} ({sig.display()})" for symbol, sig in nearest)
        logger.info("No entry signals this cycle. Closest: %s", summary)


def run_cycle(data_provider, position_tracker: PositionTracker, risk_manager: RiskManager,
              execution_client: GmxExecutionClient):
    open_positions = position_tracker.get_open_positions()
    account_equity_usd = position_tracker.get_account_equity_usd(open_positions)

    position_regimes = classify_position_regimes(data_provider, open_positions)

    closed_symbols = handle_exits(data_provider, position_tracker, execution_client, open_positions, position_regimes)

    # closed_symbols is used just to avoid re-entering the same symbol
    # this same cycle — in debug_mode nothing actually closed on-chain,
    # but we still don't want to "enter" a symbol we just logged an exit for.
    signal_count, error_count, signal_results, decisions = handle_entries(
        data_provider, risk_manager, execution_client, open_positions, account_equity_usd, closed_symbols,
        position_regimes,
    )

    try:
        write_status(STATUS_PATH, open_positions, signal_results, decisions, account_equity_usd)
    except Exception:
        logger.exception("Failed to write status snapshot for gmx_charts (non-fatal)")

    logger.info(
        "Cycle complete: %d open position(s), %d symbol(s) scanned, "
        "%d entry signal(s), %d error(s), equity=$%.2f",
        len(open_positions), len(settings.market.symbols),
        signal_count, error_count, account_equity_usd,
    )


def main():
    settings.validate()
    logger.info(
        "Starting bot | chain=%s debug_mode=%s symbols=%s",
        settings.chain.chain, settings.execution.debug_mode, settings.market.symbols,
    )

    data_provider = build_data_provider()
    position_tracker = PositionTracker(settings.chain)
    risk_manager = RiskManager(settings.risk)
    execution_client = GmxExecutionClient(settings.chain, settings.execution)

    while True:
        try:
            run_cycle(data_provider, position_tracker, risk_manager, execution_client)
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down.")
            return
        except Exception:
            logger.exception("Unhandled error in trading cycle")
        try:
            time.sleep(settings.execution.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down.")
            return


if __name__ == "__main__":
    main()
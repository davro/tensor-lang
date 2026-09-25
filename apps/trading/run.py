#!/usr/bin/env python3
"""
Runs gmx_bot and gmx_charts together, so you don't need two terminals.

Usage (from anywhere — paths are resolved relative to this script):
    python3 apps/trading/run_all.py

Each subprocess's output is streamed here with a short prefix ([bot] /
[chart]) so both logs interleave legibly in one terminal instead of two
separate scrollback buffers. Ctrl+C stops both cleanly. If either
process exits on its own (e.g. a crash), the other is stopped too
rather than silently continuing with only half the pair running.

This script only starts/stops the two existing entry points
(gmx_bot/main.py and gmx_charts/tools/chart_server.py) as separate
subprocesses — it does not merge them, share their Python state, or
change how either one runs on its own.
"""
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # apps/trading/
BOT_DIR = ROOT / "gmx_bot"
CHARTS_TOOLS_DIR = ROOT / "gmx_charts" / "tools"

PROCS = []          # every subprocess we started, so shutdown() can reach them
_shutting_down = False  # guards against shutdown() re-entering itself


def stream_output(proc: subprocess.Popen, prefix: str) -> None:
    """Reads a subprocess's stdout line by line and re-prints it with a
    prefix. Runs in its own thread per subprocess since proc.stdout is
    blocking and we're watching two of these at once."""
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="")


def start(cmd, cwd: Path, prefix: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    PROCS.append(proc)
    threading.Thread(target=stream_output, args=(proc, prefix), daemon=True).start()
    return proc


def shutdown(reason: str = "Stopping") -> None:
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    print(f"\n{reason} — stopping bot and chart server...")
    for proc in PROCS:
        if proc.poll() is None:
            proc.terminate()
    for proc in PROCS:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0)


def handle_signal(_signum, _frame) -> None:
    shutdown("Received interrupt")


def main() -> None:
    bot_entry = BOT_DIR / "main.py"
    chart_entry = CHARTS_TOOLS_DIR / "chart_server.py"
    if not bot_entry.exists():
        sys.exit(f"Can't find {bot_entry} — check apps/trading/gmx_bot exists next to this script.")
    if not chart_entry.exists():
        sys.exit(f"Can't find {chart_entry} — check apps/trading/gmx_charts exists next to this script.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # cwd is set to each app's own directory (not ROOT) so relative
    # paths inside each app — gmx_bot's .env via load_dotenv(), and
    # whatever gmx_charts' data/ paths assume — resolve exactly as they
    # would running `cd gmx_bot && python3 main.py` by hand.
    start([sys.executable, "main.py"], cwd=BOT_DIR, prefix="bot")
    start([sys.executable, "chart_server.py"], cwd=CHARTS_TOOLS_DIR, prefix="chart")

    print("Both running (bot + chart server). Ctrl+C to stop both.\n")

    while True:
        time.sleep(0.5)
        for proc in PROCS:
            if proc.poll() is not None:
                shutdown(f"'{' '.join(proc.args)}' exited with code {proc.returncode}")


if __name__ == "__main__":
    main()

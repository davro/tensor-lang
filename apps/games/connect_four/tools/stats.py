"""
Tracks cumulative game results across play.py sessions, persisted at
apps/games/connect_four/data/play_stats.json.

Kept deliberately separate from data/meta.json, which describes the
self-play TRAINING dataset generate_data.py produced (a one-time batch
job) — this file instead accumulates every game actually played
through tools/play.py, across as many runs as you like, broken down by
which mode was played (human vs AI, or AI-vs-AI self-play) and, for
human-vs-AI games, whether the human or the AI won.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # apps/
from tlkit import chunked_runner  # noqa: E402

STATS_FILENAME = "play_stats.json"

_EMPTY_MODE_TOTALS = {"games": 0, "red_wins": 0, "blue_wins": 0, "draws": 0}


def _stats_path() -> Path:
    repo_root = chunked_runner.find_repo_root()
    return repo_root / "apps" / "games" / "connect_four" / "data" / STATS_FILENAME


def load_stats() -> dict:
    path = _stats_path()
    if not path.exists():
        return {
            "games_played": 0,
            "red_wins": 0,
            "blue_wins": 0,
            "draws": 0,
            "by_mode": {
                "human_vs_ai": dict(_EMPTY_MODE_TOTALS),
                "ai_self_play": dict(_EMPTY_MODE_TOTALS),
            },
        }
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # Corrupt/partial file (e.g. killed mid-write) — don't crash the
        # game over a stats file; start fresh rather than lose the game.
        return {
            "games_played": 0,
            "red_wins": 0,
            "blue_wins": 0,
            "draws": 0,
            "by_mode": {
                "human_vs_ai": dict(_EMPTY_MODE_TOTALS),
                "ai_self_play": dict(_EMPTY_MODE_TOTALS),
            },
        }


def save_stats(stats: dict) -> None:
    path = _stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def record_result(winner, mode: str) -> dict:
    """
    winner: 0 (red/player 1 won), 1 (blue/player 2 won), or None (draw)
    mode:   "human_vs_ai" or "ai_self_play"
    Returns the updated stats dict (also persisted to disk).
    """
    if mode not in ("human_vs_ai", "ai_self_play"):
        raise ValueError(f"unknown mode {mode!r}")

    stats = load_stats()
    stats["games_played"] += 1
    if winner == 0:
        stats["red_wins"] += 1
    elif winner == 1:
        stats["blue_wins"] += 1
    else:
        stats["draws"] += 1

    mode_totals = stats["by_mode"].setdefault(mode, dict(_EMPTY_MODE_TOTALS))
    mode_totals["games"] += 1
    if winner == 0:
        mode_totals["red_wins"] += 1
    elif winner == 1:
        mode_totals["blue_wins"] += 1
    else:
        mode_totals["draws"] += 1

    save_stats(stats)
    return stats

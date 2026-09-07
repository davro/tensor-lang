"""
The "TensorLang player": wraps apps/games/tic_tac_toe/infer.tl behind a
plain `choose_move(board, mover)` function so tools/play.py doesn't need
to know anything about subprocesses, .npy files, or TensorLang at all.

Board representation used by the REST of this app (play.py, etc.):
    a list of 9 ints, index 0..8 = top-left..bottom-right, row-major.
    0 = empty, 1 = X, -1 = O.

The network itself only ever sees the perspective-relative encoding
(+1 = mover's own mark, -1 = opponent's, 0 = empty) — see encode_board().
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# from apps/games/tic_tac_toe/tools/agent.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP = "apps/games/tic_tac_toe"
STATS_FILENAME = "self_play_stats.json"


def encode_board(board, mover):
    """board: list[9] of 0/1/-1. mover: 1 or -1 (whose turn it is).
    Returns (1, 9) float32, mover's marks as +1, opponent's as -1.
    """
    return np.array([[cell * mover for cell in board]], dtype=np.float32)


def legal_moves(board):
    return [i for i, v in enumerate(board) if v == 0]


def run_inference(board, mover, repo_root=None):
    """Writes the encoded board, runs infer.tl as a subprocess, returns
    the raw (9,) move-probability array. Raises RuntimeError on failure
    (missing trained weights, compile error, etc.) — callers should
    decide how to degrade (see choose_move's fallback).
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    infer_dir = repo_root / "cache" / "apps" / "games" / "tic_tac_toe" / "infer.tl"
    infer_dir.mkdir(parents=True, exist_ok=True)
    np.save(infer_dir / "board.npy", encode_board(board, mover))

    result = subprocess.run(
        [sys.executable, "tensorlang.py", f"{APP}/infer.tl"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    move_path = infer_dir / "move.npy"
    if result.returncode != 0 or not move_path.exists():
        raise RuntimeError(
            "infer.tl failed to produce move.npy\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return np.load(move_path).reshape(9)


def choose_move(board, mover, repo_root=None, mode="best", temperature=1.0, rng=None):
    """Returns (move, probs) for `mover` to play on `board`.

    move: a legal cell index (0-8).
    probs: the network's raw (9,) probability array (illegal cells left
        as returned by the network, un-masked) for callers that want to
        display it — e.g. a confidence overlay — or None if the network
        couldn't be reached and a random fallback move was used instead.

    Tries the trained TensorLang network first; if that fails for any
    reason (weights not trained yet, a GPU hiccup, etc.) falls back to a
    random legal move so the game stays playable rather than crashing —
    printing a warning either way, since a silent fallback would hide a
    real problem the way HANDOVER.md's gotcha #2 did.

    mode:
        "best"   - always the network's top legal move (deterministic).
                   Used for the AI's moves when playing against a human,
                   so it's always playing as strong as it can.
        "sample" - weighted-random choice over legal moves, weighted by
                   the network's own probabilities (raised to
                   1/temperature first — lower temperature sharpens
                   toward "best", higher flattens toward uniform). Used
                   for self-play, since the network is otherwise fully
                   deterministic and "best" vs "best" would replay the
                   exact same game every single time.
    """
    moves = legal_moves(board)
    if not moves:
        raise ValueError("choose_move called with no legal moves left")

    try:
        probs = run_inference(board, mover, repo_root)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"[agent] TensorLang inference failed, falling back to a random move: {e}")
        return int((rng or np.random.default_rng()).choice(moves)), None

    # Mask illegal (occupied) cells before deciding anything else.
    masked = np.full(9, -np.inf, dtype=np.float32)
    masked[moves] = probs[moves]

    if mode == "best":
        return int(np.argmax(masked)), probs

    if mode == "sample":
        rng = rng or np.random.default_rng()
        legal_probs = np.clip(probs[moves], 1e-8, None) ** (1.0 / max(temperature, 1e-3))
        legal_probs = legal_probs / legal_probs.sum()
        return int(rng.choice(moves, p=legal_probs)), probs

    raise ValueError(f"choose_move: unknown mode {mode!r}, expected 'best' or 'sample'")


def load_self_play_stats(repo_root=None):
    """Returns {"1": x_wins, "-1": o_wins, "0": draws} (string keys, since
    that's what round-tripping through JSON gives you) persisted from
    previous self-play sessions, or all-zero if there's no file yet.
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    stats_path = repo_root / "cache" / "apps" / "games" / "tic_tac_toe" / STATS_FILENAME
    if not stats_path.exists():
        return {"1": 0, "-1": 0, "0": 0}
    try:
        return json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"1": 0, "-1": 0, "0": 0}


def save_self_play_stats(tally, repo_root=None):
    """tally: {1: x_wins, -1: o_wins, 0: draws} (int keys, as used in
    play.py's in-memory tally). Written after every completed self-play
    round so a long-running session (or one you fall asleep during)
    doesn't lose its scoreboard when the window closes.
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    stats_path = repo_root / "cache" / "apps" / "games" / "tic_tac_toe" / STATS_FILENAME
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({str(k): v for k, v in tally.items()}))

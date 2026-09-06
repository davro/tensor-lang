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
import subprocess
import sys
from pathlib import Path

import numpy as np

# from apps/games/tic_tac_toe/tools/agent.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP = "apps/games/tic_tac_toe"


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


def choose_move(board, mover, repo_root=None):
    """Returns a legal cell index (0-8) for `mover` to play on `board`.

    Tries the trained TensorLang network first; if that fails for any
    reason (weights not trained yet, a GPU hiccup, etc.) falls back to a
    random legal move so the game stays playable rather than crashing —
    printing a warning either way, since a silent fallback would hide a
    real problem the way HANDOVER.md's gotcha #2 did.
    """
    moves = legal_moves(board)
    if not moves:
        raise ValueError("choose_move called with no legal moves left")

    try:
        probs = run_inference(board, mover, repo_root)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"[agent] TensorLang inference failed, falling back to a random move: {e}")
        return int(np.random.default_rng().choice(moves))

    # Mask illegal (occupied) cells, then take the best of what's left.
    masked = np.full(9, -np.inf, dtype=np.float32)
    masked[moves] = probs[moves]
    return int(np.argmax(masked))

"""
The "TensorLang player": wraps apps/games/connect_four/infer.tl behind a
plain `choose_move(game, mode)` function so tools/play.py doesn't need to
know anything about subprocesses, .npy files, or TensorLang at all.

Falls back to the plain-Python HeuristicSolver (engine_solver.py) if
inference fails for any reason (no promoted weights yet, a GPU hiccup,
a compile error) — same degrade-rather-than-crash philosophy as
tic_tac_toe's and 2048's agent.py. The fallback here is a genuinely
strong solver (alpha-beta search), not a random move, so the game stays
playable and reasonably tough even before you've ever trained a network.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

# from apps/games/connect_four/tools/agent.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # connect_four/
from tools.engine_bitboard import Game, WIDTH, HEIGHT  # noqa: E402
from tools.engine_solver import HeuristicSolver  # noqa: E402

APP = "apps/games/connect_four"

_fallback_solver = HeuristicSolver(max_depth=8, time_limit=2.0)


def encode_board(game: Game) -> np.ndarray:
    """(1, 42) float32: +1 mover's own disc, -1 opponent's, 0 empty.
    Must match tools/generate_data.py's encode_board exactly.
    """
    mover_id = game.current_player + 1
    grid = game.grid()
    flat = np.zeros(WIDTH * HEIGHT, dtype=np.float32)
    i = 0
    for row in grid:
        for cell in row:
            flat[i] = 0.0 if cell == 0 else (1.0 if cell == mover_id else -1.0)
            i += 1
    return flat.reshape(1, -1)


def run_inference(game: Game, repo_root=None):
    """Writes the encoded board, runs infer.tl as a subprocess, returns
    (policy_probs (7,), value (float)). Raises RuntimeError on failure —
    callers should decide how to degrade (see choose_move's fallback).
    """
    repo_root = repo_root or chunked_runner.find_repo_root()
    infer_dir = repo_root / "cache" / "apps" / "games" / "connect_four" / "infer.tl"
    infer_dir.mkdir(parents=True, exist_ok=True)
    np.save(infer_dir / "board.npy", encode_board(game))

    result = subprocess.run(
        [sys.executable, "tensorlang.py", f"{APP}/infer.tl"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    policy_path = infer_dir / "policy.npy"
    value_path = infer_dir / "value.npy"
    if result.returncode != 0 or not policy_path.exists() or not value_path.exists():
        raise RuntimeError(
            "infer.tl failed to produce policy.npy/value.npy\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    policy = np.load(policy_path).reshape(WIDTH)
    # value head has 2 output units, trained as [value, -value] (see
    # train.tl's "REAL BUG #3" note) -- averaging (col0 - col1)/2 gives
    # a little extra robustness if the two drift apart during training
    raw_value = np.load(value_path).reshape(2)
    value = float((raw_value[0] - raw_value[1]) / 2.0)
    return policy, value


def choose_move(game: Game, repo_root=None, mode="best", temperature=1.0, rng=None):
    """Returns (col, info) for the player to move on `game`.

    info is a dict with whichever of these the source provided:
      {"source": "network", "policy": (7,) probs, "value": float}
      {"source": "solver_fallback", "score": int}

    mode:
        "best"   - always the top legal move (deterministic). Used for
                   the AI's moves against a human.
        "sample" - weighted-random legal move, weighted by the network's
                   own policy probabilities (raised to 1/temperature).
                   Used for self-play-style demo games so "best" vs
                   "best" doesn't replay the same game every time. Only
                   applies to the network path; the solver fallback
                   always plays its single best move regardless of mode.

    TACTICAL SAFETY NET: before consulting the network (or the solver
    fallback) at all, this checks for an immediate winning move or a
    forced block using the same bitboard primitives HeuristicSolver
    itself relies on (game.board.winning_moves() /
    opponent_winning_moves() — O(7), exact, not a heuristic). A modest
    MLP trained on a few thousand self-play positions can absolutely
    still miss a one-move win or fail to block one — that's exactly the
    kind of "obviously wrong" mistake more training data narrows down
    but never strictly guarantees away, whereas this check makes it
    structurally impossible regardless of network quality. Returns
    immediately with source "tactical_override" if triggered; the
    network/solver is never even called. If there are 2+ simultaneous
    winning threats to block, no single move stops both — that position
    is already lost, so this intentionally does NOT special-case it and
    falls through to the network/solver's own best (if losing) choice.
    """
    legal = game.legal_moves()
    if not legal:
        raise ValueError("choose_move called with no legal moves left")

    wins = game.board.winning_moves()
    if wins:
        return wins[0], {"source": "tactical_override", "reason": "immediate_win"}
    must_block = game.board.opponent_winning_moves()
    if len(must_block) == 1:
        return must_block[0], {"source": "tactical_override", "reason": "forced_block"}

    try:
        policy, value = run_inference(game, repo_root)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
        print(f"[agent] TensorLang inference failed, falling back to the solver: {e}")
        col, score = _fallback_solver.best_move(game.board)
        return col, {"source": "solver_fallback", "score": score}

    masked = np.full(WIDTH, -np.inf, dtype=np.float32)
    masked[legal] = policy[legal]

    if mode == "best":
        return int(np.argmax(masked)), {"source": "network", "policy": policy, "value": value}

    if mode == "sample":
        rng = rng or np.random.default_rng()
        legal_probs = np.clip(policy[legal], 1e-8, None) ** (1.0 / max(temperature, 1e-3))
        legal_probs = legal_probs / legal_probs.sum()
        col = int(rng.choice(legal, p=legal_probs))
        return col, {"source": "network", "policy": policy, "value": value}

    raise ValueError(f"choose_move: unknown mode {mode!r}, expected 'best' or 'sample'")

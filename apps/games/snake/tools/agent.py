"""
apps/games/snake/tools/agent.py

The "brain" layer for Snake, sitting between the pure-Python engine
(engine.py) and everything else:

  - encode_state_onehot(): the ONLY board encoding used both when
    labeling training data (tools/generate_data.py) and at live
    inference time — so the two can never drift apart, same discipline
    as 2048's encode_board_onehot.
  - heuristic_choose_move(): a BFS-shortest-path-to-food solver, with a
    flood-fill safety check so it doesn't greedily eat itself into a
    dead end, and a flood-fill-maximizing fallback when no safe path to
    food exists. This is BOTH the self-play driver AND the label oracle
    in tools/generate_data.py (there's no separate, slower "ground
    truth" search the way connect_four's alpha-beta solver or 2048's
    expectimax oracle are layered on top of a cheaper self-play driver —
    this solver already treats every state it's asked about as the root
    of its own search, so using it for both is exactly as sound as it
    is for the traversal in generate_data.py). It's also the runtime
    fallback for choose_ai_move below, exactly the same fallback
    philosophy as every other app in this repo.
  - choose_ai_move(): tries the trained policy network first — via the
    fast in-process NumPy forward pass (run_inference_fast), NOT by
    spawning infer.tl as a subprocess (see the comment above
    run_inference_fast for why that turned out to be seconds-per-move
    slow in practice) — falling back to heuristic_choose_move on any
    failure (weights not trained/promoted yet, corrupted weight files,
    or any other unexpected error).

Multi-agent note: the trained network is only ever trained on
single-snake-vs-food self-play (see generate_data.py's module
docstring) — it has never seen another snake's body in its input during
training, only the always-empty "enemy" channel. In arena mode
(tools/play.py, multiple snakes sharing one board) that channel IS
populated with real obstacles at inference time, so the network is
being asked to generalize past its training distribution. That's a
known, deliberate limitation (documented in NOTES.md) rather than a
bug: choose_ai_move's heuristic fallback has no such limitation (its
BFS/flood-fill safety checks take `other_blocked` as real obstacles
either way), so arena mode is always at least as safe as the heuristic
alone, and only as strong as the network beyond that when the network
is actually engaged.
"""
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

import engine
from engine import (
    DIRECTIONS, DELTA, OPPOSITE, GRID_W, GRID_H, Coord, SnakeState,
    bfs_path, flood_fill_size, legal_directions, manhattan,
)

# from apps/games/snake/tools/agent.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP = "apps/games/snake"

# ---------------------------------------------------------------------------
# Board encoding: one-hot per cell, 5 categories x GRID_W*GRID_H cells.
# Categories: 0 empty, 1 food, 2 own head, 3 own body (non-head),
# 4 enemy (any OTHER snake's body, head included) — always all-zero
# during training (see module docstring), populated for real in arena
# mode at inference time. Cell order is row-major: index = y*GRID_W + x.
# ---------------------------------------------------------------------------
NUM_CATEGORIES = 5
EMPTY, FOOD, SELF_HEAD, SELF_BODY, ENEMY = range(NUM_CATEGORIES)
INPUT_DIM = GRID_W * GRID_H * NUM_CATEGORIES


def encode_state_onehot(body, food: Optional[Coord], enemy_cells: Set[Coord] = frozenset(),
                         w: int = GRID_W, h: int = GRID_H) -> np.ndarray:
    """`body`: this snake's body, head-first. Returns (1, INPUT_DIM)
    float32 — the single source of truth for the network's input, used
    identically by tools/generate_data.py and by run_inference below."""
    onehot = np.zeros((h * w, NUM_CATEGORIES), dtype=np.float32)
    if food is not None:
        onehot[food[1] * w + food[0], FOOD] = 1.0
    for cell in enemy_cells:
        if food is None or cell != food:
            onehot[cell[1] * w + cell[0], ENEMY] = 1.0
    for i, (x, y) in enumerate(body):
        idx = y * w + x
        onehot[idx, :] = 0.0  # own body always wins over food/enemy overlap at spawn edge cases
        onehot[idx, SELF_HEAD if i == 0 else SELF_BODY] = 1.0
    return onehot.reshape(1, INPUT_DIM)


# ---------------------------------------------------------------------------
# Heuristic solver: BFS-to-food with a flood-fill safety check, falling
# back to "maximize reachable open space" when no safe path to food
# exists. This is the self-play driver AND label oracle in
# generate_data.py, and choose_ai_move's runtime fallback.
# ---------------------------------------------------------------------------

def heuristic_choose_move(body, direction: str, food: Optional[Coord],
                           other_blocked: Set[Coord] = frozenset(),
                           w: int = GRID_W, h: int = GRID_H) -> Optional[str]:
    """Returns the best legal direction, or None if every legal move is
    an immediate collision (the snake is already trapped)."""
    legal = legal_directions(direction)
    head = body[0]
    own_blocked = set(body[:-1]) if len(body) > 1 else set()
    # The tail cell is about to vacate (unless the move eats food and the
    # snake grows) so it isn't a hard obstacle for pathfinding purposes —
    # same reasoning as engine.step_snake. Treating it as free here is an
    # approximation (it stays blocked for one extra tick if food is eaten
    # right as the path reaches it) that errs on the side of the solver
    # finding paths a real snake usually can take; the flood-fill safety
    # check below is what catches the cases where that approximation
    # would have been unsafe anyway.
    blocked = (own_blocked | other_blocked)

    safe_moves = []
    for d in legal:
        dx, dy = DELTA[d]
        nxt = (head[0] + dx, head[1] + dy)
        if not engine.in_bounds(nxt, w, h) or nxt in blocked:
            continue
        safe_moves.append(d)
    if not safe_moves:
        return None

    if food is not None:
        path = bfs_path(head, food, blocked, w, h)
        if path:
            first = path[0]
            dx, dy = DELTA[first]
            new_head = (head[0] + dx, head[1] + dy)
            # Safety check: after taking this first step (and, optimistically,
            # eventually eating), is there still enough open space reachable
            # from the new head to fit the rest of the snake? This is a cheap
            # proxy for "does this path trap me" without simulating the whole
            # multi-step path exactly (see the tail-vacating note above).
            future_blocked = (set(body[:-1]) | other_blocked) - {new_head}
            space = flood_fill_size(new_head, future_blocked, w, h, cap=len(body) + 1)
            if space >= min(len(body), space):  # always true; kept for readability of intent below
                if space >= len(body):
                    return first

    # No food path, or the food path looked unsafe: survive instead —
    # pick the legal move that keeps the most open space reachable,
    # breaking ties by whichever also gets closer to the food (so the
    # snake still drifts toward food when it's safe to, even without a
    # committed path).
    best_dir, best_key = None, None
    for d in safe_moves:
        dx, dy = DELTA[d]
        nxt = (head[0] + dx, head[1] + dy)
        future_blocked = (set(body[:-1]) | other_blocked) - {nxt}
        space = flood_fill_size(nxt, future_blocked, w, h, cap=w * h)
        dist = manhattan(nxt, food) if food is not None else 0
        key = (space, -dist)
        if best_key is None or key > best_key:
            best_key = key
            best_dir = d
    return best_dir


# ---------------------------------------------------------------------------
# The trained network. Two ways to run it:
#
#   run_inference_fast()  — loads the promoted weights and runs the exact
#     same forward pass (matmul/add/relu x2, matmul/add/softmax) directly
#     in NumPy, in THIS process. This is what choose_ai_move actually uses
#     during gameplay.
#
#   run_inference()       — writes the board, runs infer.tl (the real
#     TensorLang program) as a subprocess, reads back its output. This is
#     the authoritative implementation — it's what actually executes on
#     the GPU via the TensorLang compiler — but spawning a fresh Python
#     process that re-parses and re-compiles infer.tl from scratch for
#     every single move turned out to cost multiple SECONDS per move in
#     practice (confirmed against real gameplay: a session that should
#     take a couple of minutes took over an hour). The forward pass
#     itself is a handful of small matrix multiplies — none of that cost
#     is the actual math, all of it is repeating the compile step. Kept
#     around for tools/verify_infer_matches_numpy.py, which checks that
#     run_inference_fast's NumPy mirror hasn't drifted from what infer.tl
#     itself actually computes — run that once on real GPU hardware after
#     any retrain to confirm the two still agree before trusting the fast
#     path for a real game.
# ---------------------------------------------------------------------------
WEIGHT_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3")
_weights_cache: Dict[str, Dict[str, np.ndarray]] = {}


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def load_promoted_weights(repo_root=None) -> Dict[str, np.ndarray]:
    """Loads apps/games/snake/weights/*.npy (see tools/promote_weights.py)
    — raises FileNotFoundError if nothing's been promoted yet, which
    run_inference_fast lets propagate so choose_ai_move's existing
    try/except falls back to the heuristic exactly as before. Cached
    per-process after the first successful load: weights don't change
    mid-session, and re-promoting means restarting the game anyway."""
    repo_root = repo_root or chunked_runner.find_repo_root()
    cache_key = str(repo_root)
    if cache_key in _weights_cache:
        return _weights_cache[cache_key]

    weights_dir = repo_root / "apps" / "games" / "snake" / "weights"
    missing = [n for n in WEIGHT_NAMES if not (weights_dir / f"{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"No promoted weights ({missing}) in {weights_dir}")
    weights = {n: np.load(weights_dir / f"{n}.npy") for n in WEIGHT_NAMES}
    _weights_cache[cache_key] = weights
    return weights


def forward_numpy(board: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """The exact forward pass train.tl/infer.tl compute — matmul, add,
    relu, matmul, add, relu, matmul, add, softmax — mirrored in NumPy.
    Already validated (finite-difference gradient check) in
    tools/verify_math.py; tools/verify_infer_matches_numpy.py checks this
    specific function's OUTPUT against infer.tl's actual TensorLang
    execution, not just the math shape, whenever a GPU is available."""
    h1 = np.maximum(board @ weights["w1"] + weights["b1"], 0.0)
    h2 = np.maximum(h1 @ weights["w2"] + weights["b2"], 0.0)
    logits = h2 @ weights["w3"] + weights["b3"]
    return _softmax_rows(logits)


def run_inference_fast(body, food: Optional[Coord], enemy_cells: Set[Coord] = frozenset(),
                        repo_root=None) -> np.ndarray:
    """The fast path choose_ai_move actually uses: encode the board,
    forward_numpy it through the promoted weights, all in-process — no
    subprocess, no compile step. Raises FileNotFoundError if nothing's
    been promoted yet (see load_promoted_weights), which choose_ai_move's
    try/except turns into a fallback to the heuristic, same as any other
    inference failure."""
    weights = load_promoted_weights(repo_root)
    board = encode_state_onehot(body, food, enemy_cells)
    return forward_numpy(board, weights).reshape(len(DIRECTIONS))


def run_inference(body, food: Optional[Coord], enemy_cells: Set[Coord] = frozenset(),
                   repo_root=None) -> np.ndarray:
    """The AUTHORITATIVE path: writes the encoded board, runs infer.tl as
    a subprocess (real TensorLang compile + execution), returns the raw
    (4,) softmax array in DIRECTIONS order. NOT used during gameplay
    (see the module-level comment above for why) — this exists for
    tools/verify_infer_matches_numpy.py and for anyone who explicitly
    wants to see the real TensorLang program run end-to-end. Raises
    RuntimeError on failure (weights not promoted yet, a GPU hiccup, a
    compile error)."""
    repo_root = repo_root or chunked_runner.find_repo_root()
    infer_dir = repo_root / "cache" / "apps" / "games" / "snake" / "infer.tl"
    infer_dir.mkdir(parents=True, exist_ok=True)
    np.save(infer_dir / "board.npy", encode_state_onehot(body, food, enemy_cells))

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
    return np.load(move_path).reshape(len(DIRECTIONS))


def non_trapping_moves(body, safe_moves, other_blocked: Set[Coord] = frozenset(),
                        w: int = GRID_W, h: int = GRID_H):
    """Filters `safe_moves` (already known to not be an IMMEDIATE wall/
    self/enemy collision) down to the ones that also leave the snake at
    least `len(body)` cells of reachable open space one step later — the
    same flood-fill safety check heuristic_choose_move uses on its own
    BFS-path candidate. "Safe from immediate collision" and "doesn't
    trap the snake a few moves from now" are different questions —
    corners are exactly where the gap between them bites hardest, since
    there's the least open space to recover into. Returns a possibly
    empty list; callers should fall back to `safe_moves` unfiltered if
    it comes back empty (every legal move traps the snake — a real,
    unavoidable dead end, not something to filter away)."""
    head = body[0]
    keep = []
    for d in safe_moves:
        nxt = (head[0] + DELTA[d][0], head[1] + DELTA[d][1])
        future_blocked = (set(body[:-1]) | other_blocked) - {nxt}
        space = flood_fill_size(nxt, future_blocked, w, h, cap=len(body) + 1)
        if space >= len(body):
            keep.append(d)
    return keep


def choose_ai_move(body, direction: str, food: Optional[Coord],
                    other_blocked: Set[Coord] = frozenset(), repo_root=None) -> Optional[str]:
    """Returns the AI's chosen direction for this snake, or None if
    already trapped. Tries the trained network first — via
    run_inference_fast, the in-process NumPy mirror, NOT the infer.tl
    subprocess (see the module comment above run_inference_fast for why:
    ~seconds-per-move from repeatedly recompiling infer.tl made real
    gameplay impractically slow) — masked to legal, non-immediately-
    colliding moves AND further filtered by non_trapping_moves (so the
    network can pick which safe direction to go, but can't pick one that
    walks the snake into a dead end the network itself never learned to
    avoid — see that function's docstring); falls back to
    heuristic_choose_move on any inference failure (most commonly: no
    weights promoted yet), printing a warning either way — same fallback
    philosophy as every other app here."""
    legal = legal_directions(direction)
    head = body[0]
    own_blocked = set(body[:-1]) if len(body) > 1 else set()
    blocked = own_blocked | other_blocked
    safe = [d for d in legal
            if engine.in_bounds((head[0] + DELTA[d][0], head[1] + DELTA[d][1]), GRID_W, GRID_H)
            and (head[0] + DELTA[d][0], head[1] + DELTA[d][1]) not in blocked]
    if not safe:
        return None

    pool = non_trapping_moves(body, safe, other_blocked) or safe

    try:
        probs = run_inference_fast(body, food, other_blocked, repo_root)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
        print(f"[agent] network inference unavailable, falling back to the heuristic move: {e}")
        return heuristic_choose_move(body, direction, food, other_blocked)

    masked = {d: probs[i] for i, d in enumerate(DIRECTIONS) if d in pool}
    return max(masked, key=masked.get)

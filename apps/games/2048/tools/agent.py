"""
The "TensorLang engine" wrapper for 2048: wraps step.tl / step_right.tl /
step_up.tl / step_down.tl behind a plain apply_move(board, direction)
function, so tools/play.py doesn't need to know anything about
subprocesses, .npy files, or TensorLang at all.

Board representation used by the REST of this app (play.py, etc.):
    a list of 16 ints, index 0..15 = top-left..bottom-right, row-major.
    0 = empty, otherwise the tile's value (2, 4, 8, ...).

There are now THREE different kinds of "which way do I move" logic in
this file, plus the reference simulator they all build on:

  - simulate_move(): a plain-Python reference implementation of ONE move,
    used for legality checks, game-over detection, and score bookkeeping.
    Fast (no subprocess), and doubles as a correctness cross-check against
    the engine's output the same way verify.py cross-checks step.tl.
  - heuristic_choose_move(): the original hand-written heuristic
    (corner-weighted board + empty-cell count). No longer the pygame UI's
    default autoplay policy (see choose_ai_move below), but still used
    as: (a) tools/generate_data.py's self-play driver for sampling
    representative board states, and (b) choose_ai_move's fallback if the
    trained network can't be reached.
  - choose_ai_move(): tries the trained TensorLang policy network first
    (tools/generate_data.py's expectimax-labeled data, via infer.tl),
    falling back to heuristic_choose_move on any failure. This is what
    the pygame UI's autoplay mode actually calls.

But the actual board mutation applied after every real move — whether the
human or the autoplay policy picked the direction — always goes through
run_move_engine(), i.e. the real GPU tensor-ops step_*.tl files.
simulate_move's board is only ever used as a fallback if the engine call
itself fails (see apply_move's docstring), mirroring tic_tac_toe's
choose_move degrade-to-random-move fallback.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# from apps/games/2048/tools/agent.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP = "apps/games/2048"
BOARD_N = 4
DIRECTIONS = ("left", "right", "up", "down")
STEP_FILE = {
    "left": "step.tl",
    "right": "step_right.tl",
    "up": "step_up.tl",
    "down": "step_down.tl",
}
BEST_SCORE_FILENAME = "best_score.json"
WIN_TILE = 2048

# ---------------------------------------------------------------------------
# Network input encoding: log2-one-hot per cell (16 cells x 16 value
# categories = 256 inputs). Category 0 = empty; category k (1..15) = tile
# value 2**k, with values >= 2**15 sharing the top category (2**16 = 65536
# tiles are already an extreme edge case on a 4x4 board). This is the
# single source of truth for the encoding — tools/generate_data.py labels
# boards.npy with it, and run_inference() below encodes the live board
# with it at inference time, so the two can never drift apart.
# ---------------------------------------------------------------------------
NUM_VALUE_CATEGORIES = 16


def cell_category(value):
    if value <= 0:
        return 0
    return min(int(math.log2(value)), NUM_VALUE_CATEGORIES - 1)


def encode_board_onehot(board):
    """board: list[16] of tile values (0 = empty, row-major). Returns
    (1, 256) float32."""
    onehot = np.zeros((BOARD_N * BOARD_N, NUM_VALUE_CATEGORIES), dtype=np.float32)
    for i, v in enumerate(board):
        onehot[i, cell_category(v)] = 1.0
    return onehot.reshape(1, BOARD_N * BOARD_N * NUM_VALUE_CATEGORIES)


# ---------------------------------------------------------------------------
# Board helpers (flat list[16] <-> (4,4) numpy, used at the agent/engine
# boundary since the engine speaks numpy and everything else in this app
# speaks a flat list, same split tic_tac_toe makes for its list[9]).
# ---------------------------------------------------------------------------

def to_grid(board):
    return np.array(board, dtype=np.float32).reshape(BOARD_N, BOARD_N)


def from_grid(grid):
    return [int(v) for v in grid.reshape(-1).tolist()]


def empty_cells(board):
    return [i for i, v in enumerate(board) if v == 0]


def new_board(rng):
    board = [0] * (BOARD_N * BOARD_N)
    spawn_tile(board, rng)
    spawn_tile(board, rng)
    return board


def spawn_tile(board, rng):
    """Mutates `board` in place, adding one 2 (90%) or 4 (10%) tile into a
    random empty cell. Returns False (and does nothing) if the board is
    already full."""
    cells = empty_cells(board)
    if not cells:
        return False
    idx = int(rng.choice(cells))
    board[idx] = 2 if rng.random() < 0.9 else 4
    return True


def has_won(board, target=WIN_TILE):
    return any(v >= target for v in board)


def is_game_over(board):
    """No empty cells AND no adjacent-equal pair in any row or column."""
    if empty_cells(board):
        return False
    grid = [board[r * BOARD_N:(r + 1) * BOARD_N] for r in range(BOARD_N)]
    for r in range(BOARD_N):
        for c in range(BOARD_N - 1):
            if grid[r][c] == grid[r][c + 1]:
                return False
    for c in range(BOARD_N):
        for r in range(BOARD_N - 1):
            if grid[r][c] == grid[r + 1][c]:
                return False
    return True


# ---------------------------------------------------------------------------
# Plain-Python reference move (fast, no subprocess). Same algorithm as
# step.tl (compact left, merge first-pair-then-carry, compact again) —
# see step.tl's header for why that specific merge order matters.
# ---------------------------------------------------------------------------

def _slide_left_row(vals):
    vals = [v for v in vals if v != 0]
    merged = []
    gained = 0
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            m = vals[i] * 2
            merged.append(m)
            gained += m
            i += 2
        else:
            merged.append(vals[i])
            i += 1
    merged += [0] * (BOARD_N - len(merged))
    return merged, gained


def simulate_move(board, direction):
    """Plain-Python reference implementation of one move. Returns
    (new_board, moved, gained) where `moved` is False if this direction
    is a no-op (nothing to slide/merge) and `gained` is the 2048 score
    gained from any merges (0 if none)."""
    grid = to_grid(board)
    if direction == "left":
        view = grid
    elif direction == "right":
        view = grid[:, ::-1]
    elif direction == "up":
        view = grid.T
    elif direction == "down":
        view = grid.T[:, ::-1]
    else:
        raise ValueError(f"unknown direction {direction!r}")

    gained = 0
    out_rows = []
    for row in view:
        new_row, row_gained = _slide_left_row([int(v) for v in row])
        out_rows.append(new_row)
        gained += row_gained
    out = np.array(out_rows, dtype=np.float32)

    if direction == "right":
        out = out[:, ::-1]
    elif direction == "up":
        out = out.T
    elif direction == "down":
        out = out[:, ::-1].T

    new_board = from_grid(out)
    moved = new_board != board
    return new_board, moved, gained


def legal_moves(board):
    """Directions that would actually change the board. Pure Python (no
    subprocess) — cheap enough to call every frame for UI hints, unlike
    run_move_engine."""
    return [d for d in DIRECTIONS if simulate_move(board, d)[1]]


# ---------------------------------------------------------------------------
# The real engine: step.tl / step_right.tl / step_up.tl / step_down.tl.
#
# Split into spawn (start the subprocess, return immediately) + collect
# (block until it's done, parse the result) so a UI caller can poll
# spawn's Popen handle instead of blocking outright — see play.py's
# run_engine_polling, which pumps pygame's event queue while waiting so a
# slow first-time CUDA kernel compile (tens of seconds — see NOTES.md)
# doesn't trip the OS's "app not responding" watchdog. run_move_engine
# below is the plain blocking version, for callers (tests, future
# generate_data.py, etc.) that don't need to keep a UI alive meanwhile.
# ---------------------------------------------------------------------------

def spawn_move_engine(board, direction, repo_root=None):
    """Writes the board and launches the direction's step_*.tl as a
    subprocess WITHOUT waiting for it. Returns (proc, step_dir); pass both
    to collect_move_engine() once proc.poll() is no longer None (or just
    to block on it immediately, same as run_move_engine does)."""
    step_file = STEP_FILE[direction]
    repo_root = repo_root or chunked_runner.find_repo_root()
    step_dir = repo_root / "cache" / "apps" / "games" / "2048" / step_file
    step_dir.mkdir(parents=True, exist_ok=True)
    np.save(step_dir / "board.npy", to_grid(board))

    proc = subprocess.Popen(
        [sys.executable, "tensorlang.py", f"{APP}/{step_file}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc, step_dir


def collect_move_engine(proc, step_dir, step_file=None):
    """Blocks until `proc` (from spawn_move_engine) finishes, then returns
    the resulting board (list[16]). Raises RuntimeError on failure —
    same contract as the old single-call run_move_engine."""
    stdout, stderr = proc.communicate()
    out_path = step_dir / "new_board.npy"
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"{step_file or step_dir.name} failed to produce new_board.npy\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    return from_grid(np.load(out_path))


def run_move_engine(board, direction, repo_root=None):
    """Writes the board, runs the direction's step_*.tl as a subprocess,
    blocks until done, returns the resulting board (list[16]). Raises
    RuntimeError on failure (compile error, GPU unavailable, etc.) —
    callers decide how to degrade, same split as tic_tac_toe's
    run_inference/choose_move. See the module comment above for the
    non-blocking spawn/collect split this is built from — prefer that
    directly if you're driving a UI that needs to stay responsive."""
    proc, step_dir = spawn_move_engine(board, direction, repo_root)
    return collect_move_engine(proc, step_dir, STEP_FILE[direction])


def apply_move(board, direction, repo_root=None, use_engine=True):
    """Returns (new_board, moved, gained, used_engine).

    Legality and score always come from the fast plain-Python
    simulate_move (an illegal move is a costless no-op — no subprocess,
    no spawned tile, exactly like real 2048). If the move IS legal and
    use_engine is True, the actual new board is recomputed by the real
    GPU tensor-ops engine (run_move_engine); on any engine failure this
    falls back to the plain-Python board instead of crashing the game,
    printing a warning either way — same fallback philosophy as
    tic_tac_toe's choose_move. If the engine succeeds but disagrees with
    the plain-Python board, that's a real bug in one of the two
    implementations, so it's also printed loudly rather than silently
    preferring one.
    """
    sim_board, moved, gained = simulate_move(board, direction)
    if not moved:
        return board, False, 0, False

    if not use_engine:
        return sim_board, True, gained, False

    try:
        engine_board = run_move_engine(board, direction, repo_root)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"[agent] {STEP_FILE[direction]} failed, falling back to the plain-Python move: {e}")
        return sim_board, True, gained, False

    if engine_board != sim_board:
        print(
            f"[agent] WARNING: {STEP_FILE[direction]}'s output disagrees with the "
            f"plain-Python reference for direction={direction!r}\n"
            f"  input        : {board}\n"
            f"  engine result: {engine_board}\n"
            f"  python result: {sim_board}"
        )
    return engine_board, True, gained, True


# ---------------------------------------------------------------------------
# Heuristic move choice (corner-weighted board + empty-cell count). Used
# by tools/generate_data.py to sample self-play states, and by
# choose_ai_move below as its fallback when the trained network isn't
# reachable.
# ---------------------------------------------------------------------------

# Classic "snake" weighting: biases high-value tiles toward one corner and
# down along a boustrophedon path, which tends to keep the board mergeable.
_CORNER_WEIGHTS = np.array([
    [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12],
    [2 ** 8,  2 ** 9,  2 ** 10, 2 ** 11],
    [2 ** 7,  2 ** 6,  2 ** 5,  2 ** 4],
    [2 ** 0,  2 ** 1,  2 ** 2,  2 ** 3],
], dtype=np.float64)

EMPTY_CELL_BONUS = 300.0


def _heuristic_score(board):
    grid = to_grid(board)
    weighted = float((grid * _CORNER_WEIGHTS).sum())
    empties = len(empty_cells(board))
    return weighted + empties * EMPTY_CELL_BONUS


def heuristic_choose_move(board):
    """Returns the best legal direction by a 1-ply heuristic search (see
    module docstring), or None if the game is already over. Uses
    simulate_move (not the engine) to stay fast enough to search every
    direction every frame — this is also why tools/generate_data.py uses
    it (rather than the much slower expectimax oracle) to drive self-play
    when sampling which board states end up in the training set."""
    best_dir, best_score = None, float("-inf")
    for d in DIRECTIONS:
        sim_board, moved, gained = simulate_move(board, d)
        if not moved:
            continue
        score = _heuristic_score(sim_board) + gained * 0.1
        if score > best_score:
            best_score = score
            best_dir = d
    return best_dir


# ---------------------------------------------------------------------------
# The trained network: infer.tl. Mirrors tic_tac_toe's
# run_inference/choose_move split exactly — see that file for the
# original pattern this was copied from.
# ---------------------------------------------------------------------------

def run_inference(board, repo_root=None):
    """Writes the log2-one-hot encoded board, runs infer.tl as a
    subprocess, returns the raw (4,) move-logit-softmax array in
    DIRECTIONS order. Raises RuntimeError on failure (weights not
    trained yet, a GPU hiccup, a compile error, etc.) — callers decide
    how to degrade (see choose_ai_move's fallback)."""
    repo_root = repo_root or chunked_runner.find_repo_root()
    infer_dir = repo_root / "cache" / "apps" / "games" / "2048" / "infer.tl"
    infer_dir.mkdir(parents=True, exist_ok=True)
    np.save(infer_dir / "board.npy", encode_board_onehot(board))

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


def choose_ai_move(board, repo_root=None):
    """Returns the pygame UI's autoplay direction for `board`, or None if
    the game is already over.

    Tries the trained TensorLang network first (tools/generate_data.py's
    expectimax-labeled policy, via infer.tl); if that fails for any
    reason (weights not trained yet, a GPU hiccup, a compile error) falls
    back to heuristic_choose_move instead of crashing the autoplay loop,
    printing a warning either way — same fallback philosophy as
    apply_move and tic_tac_toe's choose_move.
    """
    moves = legal_moves(board)
    if not moves:
        return None

    try:
        probs = run_inference(board, repo_root)
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"[agent] TensorLang inference failed, falling back to the heuristic move: {e}")
        return heuristic_choose_move(board)

    masked = {d: probs[i] for i, d in enumerate(DIRECTIONS) if d in moves}
    return max(masked, key=masked.get)


# ---------------------------------------------------------------------------
# Best-score persistence, mirroring tic_tac_toe's self_play_stats.json.
# ---------------------------------------------------------------------------

def load_best_score(repo_root=None):
    repo_root = repo_root or chunked_runner.find_repo_root()
    path = repo_root / "cache" / "apps" / "games" / "2048" / BEST_SCORE_FILENAME
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("best", 0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 0


def save_best_score(best, repo_root=None):
    repo_root = repo_root or chunked_runner.find_repo_root()
    path = repo_root / "cache" / "apps" / "games" / "2048" / BEST_SCORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best": int(best)}))

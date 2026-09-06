#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training run:

    python3 apps/games/tic_tac_toe/tools/generate_data.py

Enumerates every board state reachable by legal play (both X-to-move and
O-to-move positions, deduplicated), solves each one exactly with minimax,
and labels it with a single canonical best move. Writes:

  - apps/games/tic_tac_toe/data/boards.npy   (N, 9) float32
  - apps/games/tic_tac_toe/data/moves.npy    (N, 9) float32  (one-hot)
  - apps/games/tic_tac_toe/data/meta.json    {"num_states": N}

Board encoding (always from the perspective of the player about to move):
    +1.0 = the mover's own mark
    -1.0 = the opponent's mark
     0.0 = empty

This is the same encoding tools/agent.py uses at inference time, and the
same one main.tl's forward pass is trained on — so the network never has
to learn "am I X or O", only "what does a good position look like".

Re-run this to regenerate the dataset (it's deterministic, so re-running
produces byte-identical output).
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# from apps/games/tic_tac_toe/tools/generate_data.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP_DIR = Path("apps/games/tic_tac_toe")

WINS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]

# Preference order used only to pick ONE canonical move among ties, so the
# network is trained on a single-label target per state rather than a
# split/ambiguous one. Center, then corners, then edges is standard
# tic-tac-toe practice and gives a stable, human-plausible policy.
CELL_PREFERENCE = (4, 0, 2, 6, 8, 1, 3, 5, 7)


def winner(board):
    for a, b, c in WINS:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def legal_moves(board):
    return [i for i, v in enumerate(board) if v == 0]


@lru_cache(maxsize=None)
def minimax(board):
    """board: tuple of 9 ints, +1/-1/0, always from the mover's perspective
    (mover is always +1). Returns (value, best_move) from the mover's view:
    +1 = mover wins with best play, -1 = mover loses, 0 = draw.
    """
    w = winner(board)
    if w != 0:
        return (-1, None)  # the player who just moved made `w`; mover facing this board already lost
    moves = legal_moves(board)
    if not moves:
        return (0, None)

    best_value = None
    best_moves = []
    for m in moves:
        child = list(board)
        child[m] = 1
        # flip perspective for the opponent's reply
        flipped = tuple(-v for v in child)
        opp_value, _ = minimax(flipped)
        my_value = -opp_value
        if best_value is None or my_value > best_value:
            best_value = my_value
            best_moves = [m]
        elif my_value == best_value:
            best_moves.append(m)

    # canonical tie-break
    for pref in CELL_PREFERENCE:
        if pref in best_moves:
            return (best_value, pref)
    return (best_value, best_moves[0])


def enumerate_states():
    """DFS from the empty board, from the perspective of whoever is about
    to move, collecting every non-terminal state reached and its minimax
    best move. Recurses down EVERY legal reply (not just optimal ones) so
    the dataset also covers positions a human opponent might wander into.
    """
    seen = {}

    def visit(board):
        if board in seen:
            return
        if winner(board) != 0 or not legal_moves(board):
            return
        _, move = minimax(board)
        seen[board] = move
        for m in legal_moves(board):
            child = list(board)
            child[m] = 1
            visit(tuple(-v for v in child))

    visit((0,) * 9)
    return seen


def main():
    repo_root = chunked_runner.find_repo_root()
    data_dir = repo_root / APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    states = enumerate_states()
    boards = np.zeros((len(states), 9), dtype=np.float32)
    moves = np.zeros((len(states), 9), dtype=np.float32)
    for i, (board, move) in enumerate(states.items()):
        boards[i] = board
        moves[i, move] = 1.0

    np.save(data_dir / "boards.npy", boards)
    np.save(data_dir / "moves.npy", moves)
    (data_dir / "meta.json").write_text(json.dumps({"num_states": len(states)}, indent=2))

    print(f"Enumerated {len(states)} reachable non-terminal states")
    print(f"Wrote {data_dir/'boards.npy'} {boards.shape}, {data_dir/'moves.npy'} {moves.shape}")
    print(f"Wrote {data_dir/'meta.json'}")
    print(f"\nmain.tl's Tensor[f32, ({len(states)}, 9)] declarations must match this N.")


if __name__ == "__main__":
    main()

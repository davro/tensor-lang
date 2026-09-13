#!/usr/bin/env python3
"""
Run before the first training run (run.sh does this automatically):

    python3 apps/games/connect_four/tools/generate_data.py

Unlike tic_tac_toe's generate_data.py (which exhaustively enumerates all
4520 reachable board states and solves each one exactly), Connect Four
has on the order of 4.5 trillion legal positions — exhaustive enumeration
and exact solving from an early position are both intractable in pure
Python (see tools/engine_solver.py's ExactSolver docstring). Instead this
script SELF-PLAYS a fixed number of games using the depth-limited
alpha-beta HeuristicSolver, and records every non-terminal position
visited, labeled two ways:

  - policy target: a one-hot vector for the solver's best move,
    recomputed CLEANLY for that exact board (deterministic, no
    exploration noise) — independent of whatever move actually got
    played next (see explore below).
  - value target: the ACTUAL outcome of that self-play game (+1 win /
    -1 loss / 0 draw), from that position's mover's perspective,
    filled in only once the game finishes. This is standard
    self-play-style value labeling (the network learns "who actually
    wins from here", not a static hand-written score).

Game diversity: always taking the solver's own top move would make
self-play deterministic and repeat the same handful of games forever.
Each ply, with probability EXPLORE_EPSILON, a uniformly random *legal*
move is played instead of the solver's recommendation — the position
BEFORE that move is still labeled with the solver's real best move
(so the network never trains on "the random move was correct"), but
the game continues from wherever the random move leads, exposing the
dataset to positions the solver's own play would never wander into.

Writes:
  - apps/games/connect_four/data/boards.npy  (NUM_POSITIONS, 42) float32
  - apps/games/connect_four/data/policy.npy  (NUM_POSITIONS, 7)  float32 (one-hot)
  - apps/games/connect_four/data/value.npy   (NUM_POSITIONS, 2)  float32
    (column 0 = actual value target in [-1,1], column 1 = its negation
    -- see train.tl's "REAL BUG #3" note for why the value head has 2
    output units instead of the 1 that would be more natural)
  - apps/games/connect_four/data/meta.json

Board encoding (always from the perspective of the player about to move,
row-major, 42 cells = 6 rows x 7 columns, row 0 = top):
    +1.0 = the mover's own disc
    -1.0 = the opponent's disc
     0.0 = empty
This is the SAME encoding tools/agent.py uses at inference time.

NUM_POSITIONS here must match train.tl's declared Tensor shapes exactly
— keep the two in sync if you change DEFAULT_POSITIONS.

This was written and tested against the plain-Python game engine (see
engine_bitboard.py / engine_solver.py's own test suite), and WAS run
end-to-end in the sandbox this was built in (no GPU needed — this script
is pure Python/NumPy): the shipped apps/games/connect_four/data/*.npy
was generated with the defaults below (3000 positions, depth=6,
time_limit=0.08s/move), taking ~200s (~0.068s/position) and producing
102 self-play games (33/61/8 win/loss/draw for player 1/player 2/draw
at this shallow depth+exploration setting — not necessarily
representative of optimal play, just what this particular sample looked
like). Regenerate with more positions and/or a deeper solver
(--num-positions, --depth, --time-limit) for a larger/stronger dataset;
scale the estimate above linearly and keep train.tl's declared N in
sync with whatever you choose.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # apps/
from tlkit import chunked_runner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # connect_four/
from tools.engine_bitboard import Game, WIDTH, HEIGHT  # noqa: E402
from tools.engine_solver import HeuristicSolver  # noqa: E402

APP_DIR = Path("apps/games/connect_four")
DEFAULT_POSITIONS = 3000
EXPLORE_EPSILON = 0.12
SOLVER_DEPTH = 6
SOLVER_TIME_LIMIT = 0.08  # seconds per move — trades off label quality vs. generation time
SEED = 0


def encode_board(game: Game) -> np.ndarray:
    """(42,) float32: +1 mover's own disc, -1 opponent's, 0 empty."""
    mover_id = game.current_player + 1  # grid() uses 1/2, not 0/1
    grid = game.grid()
    flat = np.zeros(WIDTH * HEIGHT, dtype=np.float32)
    i = 0
    for row in grid:
        for cell in row:
            if cell == 0:
                flat[i] = 0.0
            elif cell == mover_id:
                flat[i] = 1.0
            else:
                flat[i] = -1.0
            i += 1
    return flat


def play_one_game(solver: HeuristicSolver, rng: np.random.Generator):
    """Returns a list of (board_encoding, policy_onehot, mover_index) for
    every non-terminal position visited, plus the game's winner (0, 1, or
    None for a draw) to backfill value targets with afterward.
    """
    game = Game()
    records = []
    while game.winner is None and not game.is_draw():
        legal = game.legal_moves()
        board_vec = encode_board(game)

        clean_col, _ = solver.best_move(game.board)
        policy = np.zeros(WIDTH, dtype=np.float32)
        policy[clean_col] = 1.0
        records.append((board_vec, policy, game.current_player))

        if rng.random() < EXPLORE_EPSILON and len(legal) > 1:
            play_col = int(rng.choice(legal))
        else:
            play_col = clean_col

        game.play(play_col)

    return records, game.winner


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--depth", type=int, default=SOLVER_DEPTH)
    parser.add_argument("--time-limit", type=float, default=SOLVER_TIME_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    repo_root = chunked_runner.find_repo_root()
    data_dir = repo_root / APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    solver = HeuristicSolver(max_depth=args.depth, time_limit=args.time_limit)

    boards, policies, mover_idx, game_ids = [], [], [], []
    game_winners = []
    games_played = 0

    while len(boards) < args.num_positions:
        records, winner = play_one_game(solver, rng)
        game_winners.append(winner)
        for board_vec, policy, mover in records:
            boards.append(board_vec)
            policies.append(policy)
            mover_idx.append(mover)
            game_ids.append(games_played)
        games_played += 1
        if games_played % 25 == 0:
            print(f"  {games_played} games played, {len(boards)}/{args.num_positions} positions")

    boards = np.stack(boards[: args.num_positions])
    policies = np.stack(policies[: args.num_positions])
    mover_idx = mover_idx[: args.num_positions]
    game_ids = game_ids[: args.num_positions]

    values = np.zeros((len(boards), 2), dtype=np.float32)
    for i, (mover, gid) in enumerate(zip(mover_idx, game_ids)):
        winner = game_winners[gid]
        if winner is None:
            v = 0.0
        elif winner == mover:
            v = 1.0
        else:
            v = -1.0
        values[i, 0] = v
        values[i, 1] = -v  # see train.tl's "REAL BUG #3" note: the value
        # head has 2 output units (not 1) so it can never produce an
        # all-ones-shaped tensor at batch=1, which crashed the compiler.
        # Column 1 is trained as the negation of column 0.

    np.save(data_dir / "boards.npy", boards)
    np.save(data_dir / "policy.npy", policies)
    np.save(data_dir / "value.npy", values)

    wins = sum(1 for w in game_winners if w == 0)
    losses = sum(1 for w in game_winners if w == 1)
    draws = sum(1 for w in game_winners if w is None)
    meta = {
        "num_positions": len(boards),
        "games_played": games_played,
        "solver_depth": args.depth,
        "solver_time_limit": args.time_limit,
        "explore_epsilon": EXPLORE_EPSILON,
        "seed": args.seed,
        "player0_wins": wins,
        "player1_wins": losses,
        "draws": draws,
        "value_label_mean": float(values[:, 0].mean()),
        "value_label_std": float(values[:, 0].std()),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {data_dir/'boards.npy'} {boards.shape}")
    print(f"Wrote {data_dir/'policy.npy'} {policies.shape}")
    print(f"Wrote {data_dir/'value.npy'} {values.shape}")
    print(f"Wrote {data_dir/'meta.json'}: {meta}")
    print(f"\ntrain.tl's Tensor[f32, ({len(boards)}, ...)] declarations must match this N.")


if __name__ == "__main__":
    main()

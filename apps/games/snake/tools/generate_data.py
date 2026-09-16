#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training run:

    python3 apps/games/snake/tools/generate_data.py

Snake's state space (every possible body-shape x food-position
combination on a 12x12 grid) is astronomically larger than tic_tac_toe's
4520 exactly-enumerable states, so — same situation 2048 and
connect_four are in — this generates labeled examples by SELF-PLAY
rather than exhaustive enumeration.

Unlike 2048 (a separate, slower expectimax oracle layered on top of a
cheap heuristic self-play driver) or connect_four (an alpha-beta solver
providing both), Snake uses ONE solver for both roles:
agent.heuristic_choose_move (BFS shortest path to food, with a
flood-fill safety check, falling back to "maximize reachable open
space" when no safe path exists — see agent.py's docstring). Every
state it's asked about is evaluated fresh as the root of its own
search, so there's no meaningful distinction here between "the policy
driving self-play" and "the oracle labeling states" the way there is
for 2048's greedy-heuristic-driver vs. deep-expectimax-oracle split.

SELF-PLAY. EXPLORE_PROB of the time, the snake takes a uniformly random
SAFE legal move instead of the solver's own recommendation, purely to
reach a wider variety of board shapes (near-walls, coiled bodies, etc.)
than a snake that only ever plays perfectly ever visits. Critically,
every recorded state is still labeled with the solver's recommendation
FOR THAT EXACT STATE (a fresh call to heuristic_choose_move), never
with whatever random move was actually taken — so exploration only
diversifies which states get seen, never what they're labeled with
(the same "DAgger-style" logic 2048's EXPLORE_PROB comment describes).

SYMMETRY AUGMENTATION. The 12x12 grid has the same dihedral symmetry
2048's 4x4 grid does, so every BASE_STATES self-play state is expanded
into all 8 of its rotations/mirror-reflections (see d4_variants).
Each of the 8 transformed boards is labeled by an INDEPENDENT, fresh
call to agent.heuristic_choose_move on that exact transformed board —
never by algebraically rotating the original label — exactly the same
"re-run the oracle per variant" discipline 2048's generate_data.py
uses. This matters even though BFS-shortest-path-to-food is itself
rotation/reflection symmetric: heuristic_choose_move's SAFETY fallback
(maximizing reachable open space when no safe food-path exists) breaks
ties by iterating a fixed (up, down, left, right) direction order, and
that fixed order does NOT rotate along with the board — so two
board+heading pairs that are geometric rotations of each other can
still land on genuinely different tie-broken moves. Re-running the
solver fresh on each variant sidesteps that entirely: every row's label
is always correct FOR THAT ROW's exact board, by construction, with no
assumption about tie-breaking equivariance needed.

Board encoding (see tools/agent.py's encode_state_onehot — the SAME
function used at inference time): one-hot per cell, 5 categories
(empty, food, own head, own body, enemy) x GRID_W*GRID_H cells. Enemy is
always the all-zero channel here — self-play is single-snake only (see
agent.py's module docstring for the resulting limitation in arena mode).

Writes:
  - apps/games/snake/data/boards.npy  (BASE_STATES*8, INPUT_DIM) float32
  - apps/games/snake/data/moves.npy   (BASE_STATES*8, 4) float32 (one-hot,
    columns in agent.DIRECTIONS order: up, down, left, right)
  - apps/games/snake/data/meta.json

Deterministic (fixed seed) — re-running reproduces byte-identical output.
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import engine  # noqa: E402
from engine import DIRECTIONS, DELTA, GRID_W, GRID_H  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP_DIR = Path("apps/games/snake")

SEED = 0
assert GRID_W == GRID_H, "d4_transforms below assumes a square grid"
N = GRID_W

MAX_MOVES_PER_EPISODE = 500
EXPLORE_PROB = 0.15
BASE_STATES = 2500
SYMMETRIES_PER_STATE = 8
TARGET_STATES = BASE_STATES * SYMMETRIES_PER_STATE


# ---------------------------------------------------------------------------
# Dihedral-group (D4) coordinate transform for a square grid. variant v
# in 0..7: v = k (0..3 rotations, no mirror) or v = 4+k (k rotations
# THEN a horizontal mirror). Only coordinates are transformed here —
# see module docstring for why each variant's LABEL is independently
# recomputed by agent.heuristic_choose_move instead of being derived
# algebraically from the original label.
# ---------------------------------------------------------------------------

def _rotate_coord(pos):
    x, y = pos
    return (N - 1 - y, x)


def _mirror_coord(pos):
    x, y = pos
    return (N - 1 - x, y)


def transform_coord(pos, k, mirror):
    for _ in range(k):
        pos = _rotate_coord(pos)
    if mirror:
        pos = _mirror_coord(pos)
    return pos


def d4_coord_variants(body, food):
    """Yields (transformed_body, transformed_food) for all 8
    dihedral-group transforms (identity included, as v=0)."""
    for v in range(8):
        k, mirror = (v, False) if v < 4 else (v - 4, True)
        t_body = [transform_coord(p, k, mirror) for p in body]
        t_food = transform_coord(food, k, mirror) if food is not None else None
        yield t_body, t_food


def infer_heading(body):
    """Recovers the direction the snake is currently facing from body
    order alone (head is always adjacent to the neck in the direction
    of its last move) — always unambiguous since every snake here has
    length >= INITIAL_LENGTH >= 2. Used so a transformed body can be
    re-labeled without having to separately track/transform a heading."""
    if len(body) < 2:
        return DIRECTIONS[0]
    (hx, hy), (nx, ny) = body[0], body[1]
    delta = (hx - nx, hy - ny)
    for d, dd in DELTA.items():
        if dd == delta:
            return d
    return DIRECTIONS[0]  # unreachable for a well-formed body


# ---------------------------------------------------------------------------
# Self-play state collection.
# ---------------------------------------------------------------------------

def collect_examples(target, rng):
    """Plays self-play episodes (agent.heuristic_choose_move, with
    EXPLORE_PROB chance of a random SAFE move instead) until `target`
    distinct (body, food) states have been recorded. Returns a
    list[(body, food)] in first-seen order — labels are computed later,
    independently per symmetry variant (see module docstring)."""
    seen: Dict = {}
    episodes = 0
    while len(seen) < target:
        episodes += 1
        snake = engine.new_snake(rng, GRID_W, GRID_H)
        food = engine.place_food(rng, snake.occupied(), GRID_W, GRID_H)
        for _ in range(MAX_MOVES_PER_EPISODE):
            if food is None:
                break
            label = agent.heuristic_choose_move(snake.body, snake.direction, food)
            if label is None:
                break  # solver itself has no safe move left; episode over

            key = (tuple(snake.body), food)
            if key not in seen:
                seen[key] = (list(snake.body), food)
                if len(seen) >= target:
                    break

            legal = engine.legal_directions(snake.direction)
            safe = []
            for d in legal:
                dx, dy = DELTA[d]
                nxt = (snake.head()[0] + dx, snake.head()[1] + dy)
                blocked = set(snake.body[:-1])
                if engine.in_bounds(nxt, GRID_W, GRID_H) and nxt not in blocked:
                    safe.append(d)
            if not safe:
                break

            direction = rng.choice(safe) if rng.random() < EXPLORE_PROB else label
            snake = engine.step_snake(snake, direction, food, w=GRID_W, h=GRID_H)
            if not snake.alive:
                break
            if snake.just_ate:
                food = engine.place_food(rng, snake.occupied(), GRID_W, GRID_H)

    print(f"Self-play: {episodes} episodes, {len(seen)} distinct (body, food) states collected")
    return list(seen.values())[:target]


def main():
    repo_root = chunked_runner.find_repo_root()
    data_dir = repo_root / APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    print(f"Collecting {BASE_STATES} self-play (body, food) states (seed={SEED})...")
    examples = collect_examples(BASE_STATES, rng)

    boards = np.zeros((TARGET_STATES, agent.INPUT_DIM), dtype=np.float32)
    moves = np.zeros((TARGET_STATES, len(DIRECTIONS)), dtype=np.float32)

    print(f"Applying {SYMMETRIES_PER_STATE} symmetries to {len(examples)} base states "
          f"= {TARGET_STATES} total rows (each independently re-labeled — see module docstring)...")
    row = 0
    skipped = 0
    for body, food in examples:
        for t_body, t_food in d4_coord_variants(body, food):
            heading = infer_heading(t_body)
            label = agent.heuristic_choose_move(t_body, heading, t_food)
            if label is None:
                # Shouldn't happen — a rotation/reflection of a state with a
                # safe move always has a safe move too — but skip defensively
                # rather than crash a run that's already well underway.
                skipped += 1
                continue
            boards[row] = agent.encode_state_onehot(t_body, t_food)
            moves[row, DIRECTIONS.index(label)] = 1.0
            row += 1

    if skipped:
        print(f"WARNING: skipped {skipped} symmetry variants with no legal move "
              f"(dataset has {row} rows, not the full {TARGET_STATES})")
        boards = boards[:row]
        moves = moves[:row]

    np.save(data_dir / "boards.npy", boards)
    np.save(data_dir / "moves.npy", moves)
    meta = {
        "num_states": row,
        "base_states": BASE_STATES,
        "symmetries_per_state": SYMMETRIES_PER_STATE,
        "grid_w": GRID_W,
        "grid_h": GRID_H,
        "input_dim": agent.INPUT_DIM,
        "explore_prob": EXPLORE_PROB,
        "seed": SEED,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {data_dir / 'boards.npy'} {boards.shape}, {data_dir / 'moves.npy'} {moves.shape}")
    print(f"Wrote {data_dir / 'meta.json'}")
    print(f"\ntrain.tl's Tensor[f32, ({row}, {agent.INPUT_DIM})] / ({row}, 4) declarations must match this N.")


if __name__ == "__main__":
    main()

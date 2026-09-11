#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training run:

    python3 apps/games/2048/tools/generate_data.py

Unlike tic_tac_toe (whose whole state space — 4520 reachable boards — can
be enumerated and solved exactly with minimax), 2048's state space is far
too large to enumerate: tiles spawn stochastically, values are unbounded,
and a full-depth exact search is intractable. So this oracle differs from
tic_tac_toe's in two ways:

  1. States to label come from SELF-PLAY, not exhaustive enumeration.
     tools/agent.py's heuristic_choose_move plays TARGET_STATES-worth of
     games (with a little randomness mixed in for diversity — see
     EXPLORE_PROB), and every board it passes through along the way is a
     candidate training example. This is deliberately the same policy
     agent.py already ships (not the exact heuristic used by the
     pygame UI's OLD autoplay, which this whole effort replaces) — it's
     just a way to see realistic mid-game board shapes, not a source of
     labels itself.
  2. Each sampled board is labeled by an EXPECTIMAX search (not exact
     minimax) over agent.py's simulate_move: alternating
       max node   (the 4 possible player moves)
       chance node (the ~2 x (num empty cells) possible tile spawns)
     down to DEPTH plies, bottoming out in agent.py's existing
     corner-weight + empty-cell heuristic (_heuristic_score) as the leaf
     evaluation. This is already a substantially stronger signal than
     that same heuristic used greedily 1-ply (which is all the OLD
     autoplay ever did) — it's now looking several moves and spawns
     ahead before committing to a direction.

Exact expectimax over ALL empty cells at every chance node, at every
level, is still too slow to run over thousands of boards in Python. Two
approximations keep it tractable (this is the "move pruning" mentioned in
NOTES.md's "next step" writeup):

  - MAX_CHANCE_SAMPLES caps how many empty cells a chance node expands.
    When there are more empties than that, a deterministic evenly-spaced
    subset is used and averaged over instead of all of them — an
    approximation of the true expectation, not the exact value, but a
    stable and reproducible one (same board always samples the same
    cells).
  - MOVE_PRUNE_TOP caps how many of a max node's legal moves get searched
    any deeper than a single ply, ranked by that same 1-ply heuristic
    first. This only applies BELOW the root: the root move that actually
    gets labeled always compares all legal directions in full.

Both are pure runtime/quality knobs — turn them up for stronger (slower)
labels, down for faster (weaker) ones.

SYMMETRY AUGMENTATION. Self-play here always drives toward the same
corner, in the same orientation (agent.heuristic_choose_move's
corner-weighting always favors the same corner of the grid) — so left on
its own, every training board would show the same "build toward this
specific corner" pattern, and a network trained on it can end up learning
"corner = top-left" as a literal, memorized feature of the board rather
than "keep your big tiles collected in *a* corner" as a general strategy.
That failure mode showed up in practice: an earlier run reached only
score ~950 with a visibly disorganized end board (duplicate values
scattered with no coherent gradient), consistent with the network
overfitting to a fixed orientation instead of generalizing.

The fix: every one of the BASE_STATES self-play boards is expanded into
all 8 of its dihedral-group symmetries (4 rotations x a left-right
mirror — see d4_transforms), and expectimax is run SEPARATELY on each of
the 8 (not just relabeled from the original's answer): agent's
corner-weighted heuristic isn't itself rotation/reflection-symmetric, so
only re-running the search on the actual transformed board keeps every
label correct for the heuristic actually in use. The dataset -- and
train.tl's hardcoded shapes -- are 8x BASE_STATES as a result. This costs
roughly 8x the labeling compute of the un-augmented approach for the same
number of underlying self-play states, but it's the highest-leverage
single change for teaching the network an orientation-independent
"corner" strategy instead of a memorized specific one.

With the defaults below (DEPTH=3, MAX_CHANCE_SAMPLES=4, MOVE_PRUNE_TOP=2,
BASE_STATES=3000 x 8 symmetries = 24000 states), expect this to take on
the order of 15-25 minutes on a modern CPU (pure Python + numpy, no GPU
needed — this never touches step.tl or the TensorLang compiler at all,
exactly like tic_tac_toe's minimax generator doesn't touch infer.tl).

Board encoding (see tools/agent.py's encode_board_onehot — this is the
SAME function used at inference time, so the two can never drift apart):
    log2-one-hot per cell, 16 cells x 16 value categories = 256 inputs.
    Category 0 = empty; category k (1..15) = tile value 2**k.

Writes:
  - apps/games/2048/data/boards.npy  (BASE_STATES*8, 256) float32
  - apps/games/2048/data/moves.npy   (BASE_STATES*8, 4)   float32 (one-hot,
    columns in agent.DIRECTIONS order: left, right, up, down)
  - apps/games/2048/data/meta.json   {"num_states", "base_states",
    "symmetries_per_state", "depth", "max_chance_samples",
    "move_prune_top"}

The final dataset size (BASE_STATES*8) is a fixed constant (not "however
many unique states self-play happens to find") specifically so train.tl's
Tensor shape declarations have a stable N to hardcode — same reasoning as
tic_tac_toe's generate_data.py printing "main.tl's declarations must
match this N", except here the generator itself guarantees the N instead
of leaving it to chance. Re-running this script is deterministic (fixed
seed throughout) and produces byte-identical output.
"""
import json
import sys
from pathlib import Path

import numpy as np

# from apps/games/2048/tools/generate_data.py, this directory holds agent.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

# from apps/games/2048/tools/generate_data.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP_DIR = Path("apps/games/2048")

SEED = 0

# --- self-play sampling (produces candidate board states) ---
MAX_MOVES_PER_GAME = 400        # generous cap; most games end well before this
EXPLORE_PROB = 0.15             # fraction of self-play moves chosen uniformly at random, for state diversity
BASE_STATES = 3000              # distinct self-play states collected, before symmetry augmentation
SYMMETRIES_PER_STATE = 8        # full dihedral group (4 rotations x mirror) — see d4_transforms
TARGET_STATES = BASE_STATES * SYMMETRIES_PER_STATE  # exact dataset size train.tl's shapes are hardcoded against

# --- expectimax oracle (labels each candidate board) ---
DEPTH = 3                       # plies of player-move lookahead
MAX_CHANCE_SAMPLES = 4          # cap on distinct empty cells expanded per chance node
MOVE_PRUNE_TOP = 2              # below the root, only fully search this many candidate moves per max node


# ---------------------------------------------------------------------------
# Symmetry augmentation: every collected board is trained on in all 8 of
# its rotations/reflections, each independently labeled — see module
# docstring for why this isn't just "rotate the label."
# ---------------------------------------------------------------------------

def d4_transforms(board):
    """Yields all 8 dihedral-group transforms of a 4x4 board (each as a
    flat list of 16, row-major) — the 4 rotations of the board, and the
    4 rotations of its left-right mirror. Each yielded board is a
    complete, independent board in its own right (agent.simulate_move
    and agent.legal_moves work on it exactly like any other board); the
    caller labels each one separately."""
    grid = np.array(board, dtype=np.int64).reshape(agent.BOARD_N, agent.BOARD_N)
    for k in range(4):
        rotated = np.rot90(grid, k)
        yield rotated.flatten().tolist()
        yield np.fliplr(rotated).flatten().tolist()


# ---------------------------------------------------------------------------
# Self-play: sample realistic mid-game board states to label. Not the
# oracle itself — see module docstring.
# ---------------------------------------------------------------------------

def collect_states(target, rng):
    """Plays self-play games (agent.heuristic_choose_move, with
    EXPLORE_PROB chance of a random legal move instead) until `target`
    distinct non-terminal board states have been seen, returning them as
    a list[list[16]] in first-seen order (so results are deterministic
    given a fixed seed, and truncating to exactly `target` is stable).
    """
    seen = {}
    games = 0
    while len(seen) < target:
        games += 1
        board = agent.new_board(rng)
        for _ in range(MAX_MOVES_PER_GAME):
            if agent.is_game_over(board):
                break
            moves = agent.legal_moves(board)
            if not moves:
                break

            key = tuple(board)
            if key not in seen:
                seen[key] = list(board)
                if len(seen) >= target:
                    break

            if rng.random() < EXPLORE_PROB:
                direction = rng.choice(moves)
            else:
                direction = agent.heuristic_choose_move(board) or rng.choice(moves)

            new_board, moved, _gained = agent.simulate_move(board, direction)
            if not moved:
                continue
            agent.spawn_tile(new_board, rng)
            board = new_board

    print(f"Self-play: {games} games, {len(seen)} distinct states collected")
    return list(seen.values())[:target]


# ---------------------------------------------------------------------------
# Expectimax oracle. See module docstring for the pruning/sampling
# approximations that keep this tractable.
# ---------------------------------------------------------------------------

def _sample_empty_cells(empties):
    if len(empties) <= MAX_CHANCE_SAMPLES:
        return empties
    stride = len(empties) / MAX_CHANCE_SAMPLES
    return [empties[int(i * stride)] for i in range(MAX_CHANCE_SAMPLES)]


def _chance_value(board, depth, cache):
    """Expected value over possible tile spawns (a deterministic,
    capped subset of empty cells when there are many — see
    MAX_CHANCE_SAMPLES)."""
    empties = agent.empty_cells(board)
    if not empties:
        # Board is full; no spawn possible. Shouldn't happen mid-search
        # since a full board with a legal move still has somewhere to
        # merge into, but fall back to a max node rather than crash.
        return _max_value(board, depth, cache)

    sampled = _sample_empty_cells(empties)
    total = 0.0
    for cell in sampled:
        for value, prob in ((2, 0.9), (4, 0.1)):
            child = list(board)
            child[cell] = value
            total += prob * _max_value(child, depth, cache)
    return total / len(sampled)


def _max_value(board, depth, cache):
    key = (tuple(board), depth)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if depth == 0 or agent.is_game_over(board):
        value = agent._heuristic_score(board)
        cache[key] = value
        return value

    moves = agent.legal_moves(board)
    if not moves:
        value = agent._heuristic_score(board)
        cache[key] = value
        return value

    # Move pruning: rank all legal moves by a cheap 1-ply heuristic
    # first, then only recurse (deeper search) into the top
    # MOVE_PRUNE_TOP of them — see module docstring. This is what keeps
    # branching bounded below the root.
    scored = []
    for d in moves:
        sim_board, _moved, gained = agent.simulate_move(board, d)
        scored.append((agent._heuristic_score(sim_board) + gained * 0.1, sim_board))
    scored.sort(key=lambda t: t[0], reverse=True)

    best = float("-inf")
    for _score, sim_board in scored[:MOVE_PRUNE_TOP]:
        v = _chance_value(sim_board, depth - 1, cache)
        if v > best:
            best = v
    cache[key] = best
    return best


def expectimax_best_move(board, depth, cache):
    """Root-level search: evaluates EVERY legal move fully (no pruning at
    the root — the whole point is picking the actual best one), each
    followed by depth-1 plies of alternating max/chance search below it.
    Returns the best direction; ties broken by agent.DIRECTIONS order,
    same canonical-single-label convention as tic_tac_toe's
    generate_data.py.
    """
    best_dir, best_value = None, float("-inf")
    for d in agent.DIRECTIONS:
        sim_board, moved, gained = agent.simulate_move(board, d)
        if not moved:
            continue
        value = _chance_value(sim_board, depth - 1, cache) + gained * 0.1
        if value > best_value:
            best_value = value
            best_dir = d
    return best_dir


def main():
    repo_root = chunked_runner.find_repo_root()
    data_dir = repo_root / APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print(f"Collecting {BASE_STATES} self-play board states (seed={SEED})...")
    states = collect_states(BASE_STATES, rng)

    boards = np.zeros((TARGET_STATES, 16 * agent.NUM_VALUE_CATEGORIES), dtype=np.float32)
    moves = np.zeros((TARGET_STATES, len(agent.DIRECTIONS)), dtype=np.float32)

    print(f"Labeling {len(states)} base states x {SYMMETRIES_PER_STATE} symmetries "
          f"= {TARGET_STATES} total, with expectimax "
          f"(depth={DEPTH}, max_chance_samples={MAX_CHANCE_SAMPLES}, "
          f"move_prune_top={MOVE_PRUNE_TOP})...")
    row = 0
    skipped = 0
    for i, board in enumerate(states):
        # A fresh cache per base state (shared only across its own 8
        # symmetry variants) rather than one cache for the whole run:
        # boards from different self-play states essentially never
        # coincide, so a global cache mostly just accumulates dead
        # weight — millions of entries, gigabytes of RAM, for no
        # measurable speedup (verified empirically; see NOTES.md).
        cache = {}
        for variant in d4_transforms(board):
            direction = expectimax_best_move(variant, DEPTH, cache)
            if direction is None:
                # Shouldn't happen — d4_transforms preserves legal-move
                # existence since it's just a relabeling of the same
                # grid — but skip defensively rather than crash a run
                # that's already tens of minutes in.
                skipped += 1
                continue
            boards[row] = agent.encode_board_onehot(variant).reshape(-1)
            moves[row, agent.DIRECTIONS.index(direction)] = 1.0
            row += 1
        if (i + 1) % 200 == 0 or (i + 1) == len(states):
            print(f"  labeled {i + 1}/{len(states)} base states ({row} rows so far)")

    if skipped:
        print(f"WARNING: skipped {skipped} symmetry variants with no legal move "
              f"(dataset has {row} rows, not the full {TARGET_STATES} — "
              f"train.tl's shapes must match {row}, not TARGET_STATES, if this happens)")
        boards = boards[:row]
        moves = moves[:row]

    np.save(data_dir / "boards.npy", boards)
    np.save(data_dir / "moves.npy", moves)
    meta = {
        "num_states": row,
        "base_states": BASE_STATES,
        "symmetries_per_state": SYMMETRIES_PER_STATE,
        "depth": DEPTH,
        "max_chance_samples": MAX_CHANCE_SAMPLES,
        "move_prune_top": MOVE_PRUNE_TOP,
        "seed": SEED,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {data_dir / 'boards.npy'} {boards.shape}, {data_dir / 'moves.npy'} {moves.shape}")
    print(f"Wrote {data_dir / 'meta.json'}")
    print(f"\ntrain.tl's Tensor[f32, ({row}, 256)] / ({row}, 4) declarations must match this N.")


if __name__ == "__main__":
    main()

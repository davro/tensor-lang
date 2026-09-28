#!/usr/bin/env python3
"""
No-GPU correctness check for engine_bitboard.py / engine_solver.py:

    python3 apps/games/connect_four/tools/verify_engine.py

Cross-checks the bitboard engine against independent, deliberately
non-bitboard reference implementations (plain column stacks for grid(),
a brute-force grid scan for win detection, naive minimax for
ExactSolver), the same methodology NOTES.md describes for the original
vendored engine's landing-row bug. Exits nonzero on any failure.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # connect_four/
from tools.engine_bitboard import Game, Board, bit, WIDTH, HEIGHT  # noqa: E402
from tools.engine_solver import HeuristicSolver, ExactSolver, SolverTimeout  # noqa: E402

FAILS = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def play_sequence(cols):
    g = Game()
    for c in cols:
        g.play(c)
    return g


def naive_grid_after(moves):
    columns = [[] for _ in range(WIDTH)]
    player = 0
    for c in moves:
        columns[c].append(player + 1)
        player = 1 - player
    grid = [[0] * WIDTH for _ in range(HEIGHT)]
    for col in range(WIDTH):
        for row_from_bottom, mark in enumerate(columns[col]):
            grid[HEIGHT - 1 - row_from_bottom][col] = mark
    return grid


def naive_winner(grid):
    """Brute-force scan, no bit tricks: returns 0/1 or None."""
    for r in range(HEIGHT):
        for c in range(WIDTH):
            v = grid[r][c]
            if v == 0:
                continue
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                if all(0 <= r + dr * i < HEIGHT and 0 <= c + dc * i < WIDTH
                       and grid[r + dr * i][c + dc * i] == v for i in range(4)):
                    return v - 1
    return None


# 1. basic mechanics ---------------------------------------------------------
g = Game()
check("fresh game: 7 legal moves", g.legal_moves() == list(range(7)))
check("fresh game: player 0 to move, no winner, no draw",
      g.current_player == 0 and g.winner is None and not g.is_draw())
g.play(3)
check("one move: turn flips, bottom-middle owned by player 1 (mark 1)",
      g.current_player == 1 and g.grid()[HEIGHT - 1][3] == 1)

g2 = Game()
for _ in range(HEIGHT):
    g2.play(0)  # alternating owners automatically, so no vertical win
check("column fills after HEIGHT discs", not g2.can_play(0) and 0 not in g2.legal_moves())
try:
    g2.play(0)
    check("playing a full column raises", False)
except ValueError:
    check("playing a full column raises", True)

# 2. alignment math, directly on bits -----------------------------------------
b = Board()
check("alignment: horizontal", b.alignment(sum(bit(c, 0) for c in range(4))))
check("alignment: vertical", b.alignment(sum(bit(2, r) for r in range(4))))
check("alignment: diagonal /", b.alignment(sum(bit(i, i) for i in range(4))))
check("alignment: diagonal \\", b.alignment(sum(bit(i, 3 - i) for i in range(4))))
check("alignment: 3 in a row is not a win", not b.alignment(sum(bit(c, 0) for c in range(3))))
check("alignment: no wrap across column boundary (col0 top + col1 bottom...)",
      not b.alignment(bit(0, 3) | bit(0, 4) | bit(0, 5) | bit(1, 0)))

# 3. legal sequences that end in each kind of win ------------------------------
h = play_sequence([0, 6, 1, 6, 2, 5, 3])
check("horizontal win", h.winner == 0 and h.grid()[HEIGHT - 1][0:4] == [1, 1, 1, 1])
v = play_sequence([0, 1, 0, 1, 0, 1, 0])
check("vertical win", v.winner == 0)
diag_up = [0, 1, 1, 2, 5, 2, 2, 3, 5, 3, 5, 3, 3]
d1 = play_sequence(diag_up)
check("diagonal / win", d1.winner == 0)
diag_down = [6 - c for c in diag_up]  # mirror image
d2 = play_sequence(diag_down)
check("diagonal \\ win (mirror)", d2.winner == 0)
try:
    d1.play(4)
    check("play() after a win raises", False)
except ValueError:
    check("play() after a win raises", True)

# 4. 500-game stress: grid AND winner vs independent references ----------------
random.seed(0)
grid_bad = winner_bad = draws = wins = 0
for _ in range(500):
    g, moves = Game(), []
    while g.winner is None and not g.is_draw():
        c = random.choice(g.legal_moves())
        g.play(c)
        moves.append(c)
        ref = naive_grid_after(moves)
        if ref != g.grid():
            grid_bad += 1
        if naive_winner(ref) != g.winner:
            winner_bad += 1
    wins += g.winner is not None
    draws += g.is_draw()
check(f"500 random games: grid() matches independent tally every move", grid_bad == 0)
check(f"500 random games: winner matches brute-force scan every move "
      f"({wins} wins, {draws} draws)", winner_bad == 0)

# 5. winning_moves / opponent_winning_moves -------------------------------------
w = play_sequence([0, 6, 1, 6, 2, 5])
check("winning_moves finds the completing column", w.board.winning_moves() == [3])
w2 = play_sequence([0, 6, 1, 6, 2])
check("opponent_winning_moves flags the forced block", w2.board.opponent_winning_moves() == [3])
check("...and the side to move has no win of its own", w2.board.winning_moves() == [])

# 6. HeuristicSolver -------------------------------------------------------------
solver = HeuristicSolver(max_depth=6, time_limit=1.0)
check("HeuristicSolver blocks a forced loss", solver.best_move(w2.board)[0] == 3)
col, score = solver.best_move(w.board)
check("HeuristicSolver takes an immediate win", col == 3 and score == 10_000)

random.seed(1)
crash = None
for _ in range(15):
    g, fast = Game(), HeuristicSolver(max_depth=4, time_limit=0.3)
    try:
        while g.winner is None and not g.is_draw():
            col, _ = fast.best_move(g.board)
            assert g.can_play(col), f"illegal move {col}"
            g.play(col)
    except Exception as e:  # noqa: BLE001
        crash = e
        break
check("15 solver-vs-solver games: terminate, no illegal moves", crash is None)

# 7. ExactSolver: independent naive minimax on a late position -------------------
def naive_minimax(moves):
    """+1 win / 0 draw / -1 loss for the player to move, plain recursion."""
    g = Game()
    for c in moves:
        g.play(c)
    if g.is_draw():
        return 0
    best = -1
    for c in g.legal_moves():
        nxt = Game()
        for m in moves + [c]:
            nxt.play(m)
        if nxt.winner is not None:
            return 1
        best = max(best, -naive_minimax(moves + [c]))
        if best == 1:
            break
    return best


random.seed(2)
late_moves = None
while late_moves is None:
    g, mv = Game(), []
    while g.winner is None and not g.is_draw() and len(mv) < 34:
        c = random.choice(g.legal_moves())
        g.play(c)
        mv.append(c)
    if g.winner is None and len(mv) == 34:
        late_moves = mv
late = play_sequence(late_moves)
exact = ExactSolver(max_nodes=2_000_000)
exact_score = exact.solve(late.board)
exact_sign = (exact_score > 0) - (exact_score < 0)
check(f"ExactSolver agrees with naive minimax on a 34-ply position "
      f"(exact={exact_score}, naive={naive_minimax(late_moves)})",
      exact_sign == naive_minimax(late_moves))

fresh_board = Game().board
try:
    ExactSolver(max_nodes=50).solve(fresh_board)
    check("ExactSolver refuses (SolverTimeout) rather than guess on an opening position", False)
except SolverTimeout:
    check("ExactSolver refuses (SolverTimeout) rather than guess on an opening position", True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")

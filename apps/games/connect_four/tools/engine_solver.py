"""
apps/games/connect_four/tools/engine_solver.py

Two solvers sharing engine_bitboard.py's primitives, per NOTES.md:

  - HeuristicSolver: depth-limited negamax + alpha-beta + a positional
    evaluation function, with iterative deepening up to max_depth cut
    short by a wall-clock time_limit. This is the one actually used —
    by tools/generate_data.py to label self-play training data, and by
    tools/agent.py as the fallback opponent whenever TensorLang
    inference isn't available. Fast enough for real-time play.

  - ExactSolver: negamax + alpha-beta + a transposition table, with no
    heuristic — it only ever returns a proven win/loss/draw score, never
    an estimate. Correspondingly, it's only practical for midgame/
    endgame analysis in pure Python; a full 42-ply solve from an early
    position is not realistic here (real bitboard connect-4 solvers that
    do this are hand-tuned C, not this). Raises SolverTimeout rather
    than pretend a heuristic guess is an exact score. Not currently
    called by any other file in this app — kept for the same reason
    NOTES.md describes the vendored original as "independently useful."

Reconstructed after the original vendored copy was lost before being
committed — see NOTES.md's "The Python engine" section and
engine_bitboard.py's module docstring.
"""
import time

from tools.engine_bitboard import WIDTH, HEIGHT, H1, Board, bit


class SolverTimeout(Exception):
    """Raised by ExactSolver.solve() when it can't finish within its node
    budget — a full exact solve from an early position is genuinely
    intractable in pure Python (see module docstring); this makes that
    failure explicit instead of silently returning a wrong score."""


# Center-out move ordering: center columns are far more likely to be part
# of a winning line, so trying them first lets alpha-beta prune the most.
_CENTER_ORDER = sorted(range(WIDTH), key=lambda c: abs(c - WIDTH // 2))

_WIN_SCORE = 10_000


def _all_windows():
    """Every set of 4 in-line cells on the board (horizontal, vertical,
    both diagonals), each as a single combined bitmask, for the
    evaluation function to popcount against."""
    windows = []
    for row in range(HEIGHT):
        for col in range(WIDTH - 3):
            windows.append(sum(bit(col + i, row) for i in range(4)))
    for col in range(WIDTH):
        for row in range(HEIGHT - 3):
            windows.append(sum(bit(col, row + i) for i in range(4)))
    for col in range(WIDTH - 3):
        for row in range(HEIGHT - 3):
            windows.append(sum(bit(col + i, row + i) for i in range(4)))
    for col in range(WIDTH - 3):
        for row in range(3, HEIGHT):
            windows.append(sum(bit(col + i, row - i) for i in range(4)))
    return windows


_ALL_WINDOWS = _all_windows()
_WINDOW_SCORE = {1: 1, 2: 10, 3: 50, 4: _WIN_SCORE}


def _evaluate(board: Board, perspective: int) -> int:
    """Heuristic score of `board` from the point of view of whichever
    player's stones are in `perspective` (positive = good for them).
    For every possible 4-cell line on the board: a line already blocked
    by both sides can never become 4-in-a-row for either, so it scores
    nothing; otherwise the side occupying it scores based on how many of
    the 4 cells they already have."""
    opponent = board.mask & ~perspective
    score = 0
    for w in _ALL_WINDOWS:
        mine = bin(w & perspective).count("1")
        theirs = bin(w & opponent).count("1")
        if mine and theirs:
            continue
        if mine:
            score += _WINDOW_SCORE[mine]
        elif theirs:
            score -= _WINDOW_SCORE[theirs]
    return score


class HeuristicSolver:
    def __init__(self, max_depth: int = 8, time_limit: float = 2.0):
        self.max_depth = max_depth
        self.time_limit = time_limit

    def best_move(self, board: Board):
        """Returns (col, score) — score positive means good for the
        player to move. Raises ValueError if there's no legal move."""
        legal = [c for c in _CENTER_ORDER if board.can_play(c)]
        if not legal:
            raise ValueError("best_move called with no legal moves")

        # Tactical shortcut: take an immediate win outright rather than
        # spend the search budget "discovering" it.
        for c in legal:
            if board.is_win_move(c):
                return c, _WIN_SCORE

        deadline = time.monotonic() + self.time_limit
        best_col, best_score = legal[0], None

        depth = 1
        try:
            while depth <= self.max_depth:
                best_col, best_score = self._search_root(board, legal, depth, deadline)
                depth += 1
        except _SearchTimeout:
            pass

        return best_col, best_score

    def _search_root(self, board, legal, depth, deadline):
        alpha, beta = float("-inf"), float("inf")
        best_col, best_score = legal[0], float("-inf")
        for c in legal:
            child = board.copy()
            child.play(c)
            score = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if score > best_score:
                best_col, best_score = c, score
            alpha = max(alpha, score)
        return best_col, best_score

    def _negamax(self, board, depth, alpha, beta, deadline):
        if time.monotonic() > deadline:
            raise _SearchTimeout()

        # Did the player who just moved (creating this exact `board`)
        # just win? `board.position` always means "whoever moves next
        # from here", so the player who just moved is position ^ mask.
        just_moved = board.position ^ board.mask
        if board.alignment(just_moved):
            # Earlier wins are worth more (prefer the fastest forced
            # win / slowest forced loss among otherwise-equal lines).
            return -(_WIN_SCORE - board.moves)

        legal = [c for c in _CENTER_ORDER if board.can_play(c)]
        if not legal:
            return 0  # board full, no winner: draw

        if depth == 0:
            return _evaluate(board, board.position)

        best = float("-inf")
        for c in legal:
            child = board.copy()
            child.play(c)
            score = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best


class _SearchTimeout(Exception):
    """Internal to HeuristicSolver — signals "iterative deepening ran out
    of time", distinct from ExactSolver's user-facing SolverTimeout."""


class ExactSolver:
    """Proven-exact negamax + alpha-beta + transposition table. No
    heuristic fallback: every score it returns is either an exact
    win/loss/draw, or it raises SolverTimeout. See module docstring for
    why this is midgame/endgame-only in practice."""

    def __init__(self, max_nodes: int = 2_000_000):
        self.max_nodes = max_nodes
        self._table = {}
        self._nodes = 0

    def solve(self, board: Board) -> int:
        """Exact score for the player to move: positive means a forced
        win, negative a forced loss, 0 a forced draw (with perfect play
        from both sides), magnitude indicating how many plies away.
        Raises SolverTimeout if max_nodes is exceeded first."""
        self._table = {}
        self._nodes = 0
        return self._negamax(board, float("-inf"), float("inf"))

    def best_move(self, board: Board):
        legal = [c for c in _CENTER_ORDER if board.can_play(c)]
        if not legal:
            raise ValueError("best_move called with no legal moves")

        for c in legal:
            if board.is_win_move(c):
                return c, _WIN_SCORE

        best_col, best_score = legal[0], float("-inf")
        for c in legal:
            child = board.copy()
            child.play(c)
            score = -self.solve(child)
            if score > best_score:
                best_col, best_score = c, score
        return best_col, best_score

    def _negamax(self, board, alpha, beta):
        self._nodes += 1
        if self._nodes > self.max_nodes:
            raise SolverTimeout(
                f"ExactSolver exceeded its {self.max_nodes}-node budget without "
                f"reaching a proven result — this position is too early/open to "
                f"solve exactly in pure Python; use HeuristicSolver instead."
            )

        just_moved = board.position ^ board.mask
        if board.alignment(just_moved):
            return -(_WIN_SCORE - board.moves)

        legal = [c for c in _CENTER_ORDER if board.can_play(c)]
        if not legal:
            return 0

        key = board.key()
        cached = self._table.get(key)
        if cached is not None:
            cached_score, cached_alpha, cached_beta = cached
            if cached_alpha <= alpha and cached_beta >= beta:
                return cached_score

        orig_alpha = alpha
        best = float("-inf")
        for c in legal:
            child = board.copy()
            child.play(c)
            score = -self._negamax(child, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        self._table[key] = (best, orig_alpha, beta)
        return best

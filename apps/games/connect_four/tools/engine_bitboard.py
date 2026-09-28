"""
apps/games/connect_four/tools/engine_bitboard.py

A complete, independently-useful Connect Four engine: Fhourstones-style
bitboard board representation (7 columns x 7 rows-per-column, including
one sentinel row per column so the four-in-a-row bit-shift checks can't
wrap across column boundaries), O(1) move legality/win checks via bit
tricks, plus a Game wrapper that tracks per-player disc ownership and
turn/winner bookkeeping for display.

Reconstructed after the original vendored copy (from a sibling
non-TensorLang project, per NOTES.md) was lost before being committed.
NOTES.md describes a bug once found and fixed here: the landing-row
count for a column must mask OFF other columns' bits before counting set
bits, not just shift and count whatever comes along. Game deliberately
avoids that whole class of bug by tracking disc ownership with a simple
incremental per-column counter (_col_heights) rather than re-deriving it
from the bitboard's shift tricks every time grid() is called.

Bit layout (the well-known scheme this style of engine is built on):
  - WIDTH columns (7), HEIGHT playable rows per column (6), plus one
    sentinel row per column (bit index HEIGHT), so H1 = HEIGHT + 1 = 7
    bits per column, 49 bits total.
  - Column c's bits occupy [c*H1, c*H1 + HEIGHT] inclusive: bit
    c*H1 + r is row r (0 = bottom) of column c; bit c*H1 + HEIGHT is
    that column's sentinel, which is 0 until the column is full and
    exists purely so a shift-based alignment check can't "see" into the
    next column.
  - `mask`: bitboard of every occupied cell (either player).
  - `position`: bitboard of the cells belonging to whichever player is
    TO MOVE NEXT (not a fixed physical player — play() flips this
    meaning every turn). The other player's stones are `position ^ mask`.
  - `key()` = `position + mask` is a unique encoding of the whole state,
    safe as a transposition-table key (the sentinel row, always 0 in
    mask, is what keeps this sum from ever aliasing two different
    states — see any writeup of this classic technique).
"""

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1  # bits per column, including the sentinel row

_COLUMN_PLAYABLE_MASK = (1 << HEIGHT) - 1  # HEIGHT low bits set


def _bottom_mask(col: int) -> int:
    return 1 << (col * H1)


def _top_mask(col: int) -> int:
    """Bit at the topmost PLAYABLE row of `col` (row HEIGHT - 1) — set
    once that column is completely full."""
    return 1 << (HEIGHT - 1 + col * H1)


def _column_mask(col: int) -> int:
    """All HEIGHT playable bits of `col` (excludes its sentinel bit)."""
    return _COLUMN_PLAYABLE_MASK << (col * H1)


def bit(col: int, row: int) -> int:
    """The single bit for (col, row), row 0 = bottom. Exposed for
    engine_solver.py's evaluation function, which needs to build
    per-window bitmasks directly."""
    return 1 << (col * H1 + row)


class Board:
    """Low-level bitboard core: legality, win detection, and search
    primitives. No turn-label/winner bookkeeping beyond `moves` — that's
    Game's job (see its docstring for why)."""

    __slots__ = ("mask", "position", "moves")

    def __init__(self, mask: int = 0, position: int = 0, moves: int = 0):
        self.mask = mask
        self.position = position
        self.moves = moves

    def copy(self) -> "Board":
        return Board(self.mask, self.position, self.moves)

    def can_play(self, col: int) -> bool:
        return (self.mask & _top_mask(col)) == 0

    def height_of(self, col: int) -> int:
        """Number of discs currently in `col` (0..HEIGHT). Correctly
        masks off other columns' bits before counting — see module
        docstring for the bug this avoids."""
        col_bits = (self.mask >> (col * H1)) & _COLUMN_PLAYABLE_MASK
        return bin(col_bits).count("1")

    def play(self, col: int) -> None:
        """Drops a disc for the player to move into `col`, then flips
        whose turn `position` represents.

        `self.mask + _bottom_mask(col)`: within one column, occupied
        bits are always contiguous from the bottom, so adding 1 at that
        column's bottom bit position carries through every already-set
        bit above it and lands exactly on the first empty one — the
        standard trick for O(1) "drop into the next free row."
        """
        self.position ^= self.mask
        self.mask |= self.mask + _bottom_mask(col)
        self.moves += 1

    def alignment(self, position: int) -> bool:
        """True if `position`'s bitboard contains 4-in-a-row, in any of
        the four directions, anywhere on the board. Public (not just
        used internally) because engine_solver.py's evaluation and
        terminal-node checks need to test arbitrary hypothetical
        positions, not just self.position."""
        # vertical (adjacent bits within a column)
        m = position & (position >> 1)
        if m & (m >> 2):
            return True
        # horizontal (adjacent columns, same row)
        m = position & (position >> H1)
        if m & (m >> (2 * H1)):
            return True
        # diagonal "/" (up-right: +1 row per +1 column)
        m = position & (position >> HEIGHT)
        if m & (m >> (2 * HEIGHT)):
            return True
        # diagonal "\" (down-right: -1 row per +1 column)
        m = position & (position >> (H1 + 1))
        if m & (m >> (2 * (H1 + 1))):
            return True
        return False

    def _position_after(self, position: int, col: int) -> int:
        """`position`'s bitboard with a disc added at `col`'s next free
        row, WITHOUT mutating self or touching `self.position`/`mask`
        directly — gravity (which row a disc lands on) is shared state,
        independent of whose hypothetical stones we're testing, which is
        exactly what lets this same helper answer both "do I win here"
        (is_win_move, using self.position) and "does the opponent win
        here" (opponent_winning_moves, using position ^ mask instead).
        """
        return position | ((self.mask + _bottom_mask(col)) & _column_mask(col))

    def is_win_move(self, col: int) -> bool:
        """True if the player to move wins immediately by playing `col`."""
        return self.alignment(self._position_after(self.position, col))

    def winning_moves(self) -> list:
        """Legal columns where the player to move wins immediately."""
        return [c for c in range(WIDTH) if self.can_play(c) and self.is_win_move(c)]

    def opponent_winning_moves(self) -> list:
        """Legal columns that, if left open, let the OPPONENT complete a
        4-in-a-row on their very next move — i.e. forced blocks."""
        opp_position = self.position ^ self.mask
        return [
            c for c in range(WIDTH)
            if self.can_play(c) and self.alignment(self._position_after(opp_position, c))
        ]

    def key(self) -> int:
        """Unique transposition-table key for the current state (see
        module docstring)."""
        return self.position + self.mask


class Game:
    """Wraps a Board with turn/winner bookkeeping and a display-friendly
    grid() — tracked directly and incrementally here (not re-derived from
    the bitboard's turn-relative position/mask trick), since Board's
    `position` means a different physical player every other move and
    re-deriving "who owns this cell" from that after the fact, especially
    around a game-ending move, is exactly the kind of easy-to-get-subtly-
    wrong bit-arithmetic NOTES.md warns about.
    """

    __slots__ = ("board", "current_player", "winner", "_owner", "_col_heights")

    def __init__(self):
        self.board = Board()
        self.current_player = 0  # 0 or 1 — player 0 moves first
        self.winner = None       # None, 0, or 1
        # HEIGHT x WIDTH, row 0 = TOP (matches tools/play.py's convention).
        # 0 = empty, 1 = player-0's disc, 2 = player-1's disc.
        self._owner = [[0] * WIDTH for _ in range(HEIGHT)]
        self._col_heights = [0] * WIDTH  # discs currently in each column

    def can_play(self, col: int) -> bool:
        return self.board.can_play(col)

    def legal_moves(self) -> list:
        return [c for c in range(WIDTH) if self.board.can_play(c)]

    def is_draw(self) -> bool:
        return self.winner is None and self.board.moves >= WIDTH * HEIGHT

    def play(self, col: int) -> None:
        if self.winner is not None or self.is_draw():
            raise ValueError("play() called on a finished game")
        if not self.board.can_play(col):
            raise ValueError(f"column {col} is full")

        won = self.board.is_win_move(col)
        mover = self.current_player

        self.board.play(col)

        row_from_bottom = self._col_heights[col]
        self._col_heights[col] += 1
        self._owner[HEIGHT - 1 - row_from_bottom][col] = mover + 1

        if won:
            self.winner = mover
        else:
            self.current_player = 1 - self.current_player

    def grid(self) -> list:
        """HEIGHT x WIDTH list of lists, row 0 = TOP, values 0 (empty),
        1, or 2 (which player's disc). Returns a fresh copy each call so
        callers can't accidentally mutate internal state."""
        return [row[:] for row in self._owner]

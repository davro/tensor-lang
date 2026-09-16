"""
apps/games/snake/tools/engine.py

The plain-Python Snake engine: grid, snake body, food, movement and
collision rules, plus the BFS-pathfinding / flood-fill primitives the
heuristic solver (agent.py) is built from. Same split as every other app
in this repo (tic_tac_toe, 2048, connect_four): TensorLang only ever runs
the neural net (train.tl / infer.tl); every game rule lives here, in
plain Python, so it can be tested fast with no GPU and no subprocess.

Board representation:
    A snake is `list[(x, y)]`, head-first (index 0 = head). Coordinates
    are (col, row), 0 <= x < GRID_W, 0 <= y < GRID_H.
    A `SnakeState` (see below) is one snake's full situation: its body,
    its current heading, whether it's alive, and its score. Food is
    shared board state (one `(x, y)` position), not part of any single
    snake's state, since multiple snakes can share one board (arena
    mode) and race for the same food.

This module is deliberately snake-count-agnostic: it only ever operates
on ONE snake's body plus a set of "blocked" cells (food excluded) that
that snake must not move into. Arena mode (tools/play.py) builds that
blocked set from every OTHER snake's body each tick; single-player mode
builds it from nothing but the snake's own body. Either way the
collision/pathfinding code below doesn't need to know how many snakes
exist.
"""
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]

GRID_W = 12
GRID_H = 12

DIRECTIONS = ("up", "down", "left", "right")
DELTA = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

INITIAL_LENGTH = 3


@dataclass
class SnakeState:
    body: List[Coord]              # head-first
    direction: str                 # current heading; never None after init
    alive: bool = True
    score: int = 0
    just_ate: bool = False         # true for exactly the tick a food was eaten, for animation hooks

    def head(self) -> Coord:
        return self.body[0]

    def occupied(self) -> Set[Coord]:
        return set(self.body)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def new_snake(rng: random.Random, w: int = GRID_W, h: int = GRID_H,
              start: Optional[Coord] = None, direction: Optional[str] = None) -> SnakeState:
    """A fresh, straight, INITIAL_LENGTH-long snake, placed with clearance
    from the walls so its very first moves are never forced into an
    instant wall collision."""
    if start is None:
        cx = rng.randint(w // 4, (3 * w) // 4)
        cy = rng.randint(h // 4, (3 * h) // 4)
        start = (cx, cy)
    if direction is None:
        direction = rng.choice(DIRECTIONS)
    dx, dy = DELTA[direction]
    body = [(start[0] - i * dx, start[1] - i * dy) for i in range(INITIAL_LENGTH)]
    return SnakeState(body=body, direction=direction)


def place_food(rng: random.Random, blocked: Set[Coord], w: int = GRID_W, h: int = GRID_H) -> Optional[Coord]:
    """A uniformly random free cell, or None if the board is completely
    full (an actual win condition — no space left to place food)."""
    free = [(x, y) for y in range(h) for x in range(w) if (x, y) not in blocked]
    if not free:
        return None
    return rng.choice(free)


# ---------------------------------------------------------------------------
# Movement / collision
# ---------------------------------------------------------------------------

def in_bounds(pos: Coord, w: int = GRID_W, h: int = GRID_H) -> bool:
    x, y = pos
    return 0 <= x < w and 0 <= y < h


def legal_directions(current_direction: str) -> Tuple[str, ...]:
    """The 3 directions that don't immediately reverse into the snake's
    own neck (all 4 only make sense before the first move, which
    new_snake already resolves by picking one)."""
    return tuple(d for d in DIRECTIONS if d != OPPOSITE[current_direction])


def step_snake(state: SnakeState, direction: str, food: Optional[Coord],
               other_blocked: Set[Coord] = frozenset(),
               w: int = GRID_W, h: int = GRID_H) -> SnakeState:
    """Advances one snake by one cell in `direction`. `other_blocked` is
    every cell occupied by OTHER snakes (empty for single-player). Head-on
    self collision or wall collision or hitting another snake's body all
    set alive=False; the returned state's body is otherwise unchanged in
    that case (a dead snake just stops, for the death animation to read
    its final resting position from).

    Eating: the tail is only dropped if the new head does NOT land on
    `food` — this is the whole growth mechanic. Score increments by 1 per
    food eaten (tools/play.py can weight this for display however it
    likes).
    """
    if not state.alive:
        return state

    dx, dy = DELTA[direction]
    head_x, head_y = state.head()
    new_head = (head_x + dx, head_y + dy)

    if not in_bounds(new_head, w, h):
        return SnakeState(state.body, direction, alive=False, score=state.score)

    # The current tail cell is about to be vacated (unless this move eats
    # food, in which case the snake grows and the tail stays put) — so it's
    # not an obstacle for the new head, exactly like real Snake.
    body_blocked = set(state.body[:-1]) if len(state.body) > 1 else set()
    if new_head in body_blocked or new_head in other_blocked:
        return SnakeState(state.body, direction, alive=False, score=state.score)

    ate = food is not None and new_head == food
    new_body = [new_head] + state.body if ate else [new_head] + state.body[:-1]
    new_score = state.score + 1 if ate else state.score
    return SnakeState(new_body, direction, alive=True, score=new_score, just_ate=ate)


# ---------------------------------------------------------------------------
# BFS pathfinding + flood-fill (used by agent.heuristic_choose_move, and
# reusable for anything else that wants "shortest safe path" or "how much
# open space is reachable from here").
# ---------------------------------------------------------------------------

def bfs_path(start: Coord, goal: Coord, blocked: Set[Coord],
             w: int = GRID_W, h: int = GRID_H) -> Optional[List[str]]:
    """Shortest sequence of directions from `start` to `goal` avoiding
    `blocked` cells, or None if unreachable. `start` itself is never
    treated as blocked even if present in `blocked`."""
    if start == goal:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for d in DIRECTIONS:
            dx, dy = DELTA[d]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited or not in_bounds(nxt, w, h) or nxt in blocked:
                continue
            if nxt == goal:
                return path + [d]
            visited.add(nxt)
            queue.append((nxt, path + [d]))
    return None


def flood_fill_size(start: Coord, blocked: Set[Coord], w: int = GRID_W, h: int = GRID_H,
                     cap: Optional[int] = None) -> int:
    """Count of cells reachable from `start` (inclusive) without crossing
    `blocked`. `cap`, if given, stops early once that many cells are
    confirmed reachable — the safety heuristic below only ever needs to
    know "is this at least as big as my body", not the exact number, so
    capping keeps it cheap on a mostly-open board."""
    if start in blocked:
        return 0
    seen = {start}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if cap is not None and len(seen) >= cap:
            return len(seen)
        for d in DIRECTIONS:
            dx, dy = DELTA[d]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or not in_bounds(nxt, w, h) or nxt in blocked:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

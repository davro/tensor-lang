#!/usr/bin/env python3
"""
Snake, played on the real TensorLang policy network — or watch one AI
snake play, or drop several AI snakes into the same arena and watch them
compete for food.

Run from the tensor-lang repo root:

    python3 apps/games/snake/tools/play.py

Start screen:
    Enter / arrow keys   play it yourself
    A                    watch one AI snake play
    M                    arena: watch several AI snakes compete
    Esc / Q              quit

In game:
    Arrow keys / WASD    steer (human mode only)
    + / -                speed up / slow down
    N                    new game (same mode)
    R                    back to the start screen
    Esc / Q              quit

Every AI decision (single-AI mode, and each snake in arena mode) goes
through tools/agent.py's choose_ai_move: the trained network's forward
pass, run directly in NumPy in-process (fast — see agent.py's module
comment for why it's NOT run by spawning infer.tl per move) first,
falling back to the BFS/flood-fill heuristic solver on any failure
(most commonly: no weights promoted yet) — same fallback philosophy as
every other app in this repo. The snake's actual MOVEMENT (collision,
growth, food) is always plain-Python engine.step_snake; only the
DIRECTION CHOICE ever goes through the network.

Arena-mode caveat: the network was only ever trained single-snake-vs-food
(see tools/agent.py's module docstring) — it has never seen another
snake's body in its input during training. Arena mode still WORKS
because choose_ai_move's blocked-cell/legality logic is always correct
regardless of training data, and the heuristic fallback has no such
blind spot either way; the network itself may just play more
conservatively than expected around other snakes until it's fine-tuned
on genuine multi-snake data (a natural next step — see NOTES.md).

Performance note: choose_ai_move runs the trained network's forward
pass directly in NumPy, in-process (see tools/agent.py's module
comment) — NOT by spawning infer.tl as a subprocess every move, which
turned out to cost multiple SECONDS per move in practice (confirmed
against a real game session: what should have taken a couple of
minutes took over an hour). Both single-AI and arena mode are fast
regardless of which driver (H) is selected; H only changes WHICH
decision algorithm plays, not how fast the game runs.
"""
import math
import random
import sys
import time
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import engine  # noqa: E402
from engine import GRID_W, GRID_H, DELTA, DIRECTIONS  # noqa: E402

CELL = 40
BOARD_W = GRID_W * CELL
BOARD_H = GRID_H * CELL
STATUS_H = 132
WINDOW_W = BOARD_W
WINDOW_H = BOARD_H + STATUS_H

BG = (24, 26, 22)
GRID_LINE = (34, 38, 32)
FOOD_COLOR = (235, 82, 82)
FOOD_GLOW = (235, 82, 82)
TEXT_COLOR = (230, 232, 224)
HINT_COLOR = (150, 155, 142)
DEATH_FLASH = (200, 40, 40)

SNAKE_PALETTE = [
    ((110, 220, 110), (40, 120, 40)),   # green head -> tail
    ((110, 170, 240), (35, 80, 150)),   # blue
    ((240, 190, 90), (150, 100, 30)),   # gold
    ((230, 120, 220), (140, 50, 130)),  # magenta
]

DEFAULT_TICK_MS = 160
TICK_MIN_MS = 40
TICK_MAX_MS = 400
TICK_STEP_MS = 15

KEY_TO_DIRECTION = {
    pygame.K_UP: "up", pygame.K_w: "up",
    pygame.K_DOWN: "down", pygame.K_s: "down",
    pygame.K_LEFT: "left", pygame.K_a: "left",
    pygame.K_RIGHT: "right", pygame.K_d: "right",
}
# Only arrow keys (not WASD) start a human game from the start screen —
# W/A/S/D collide with the A-for-AI / (future) D-for-... mode-select
# keys, so using the full KEY_TO_DIRECTION map there would swallow "A"
# before the AI-mode check ever saw it. WASD still steers once a human
# game is actually underway (see the "playing" state's key handling
# below), where there's no such collision.
ARROW_KEYS = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)

BEST_SCORE_FILENAME = "best_score.json"

# ---------------------------------------------------------------------------
# Speed slider (mouse-draggable, in addition to the +/- keys). Left =
# slowest (TICK_MAX_MS between moves), right = fastest (TICK_MIN_MS).
# ---------------------------------------------------------------------------
SLIDER_HANDLE_R = 9


def tick_to_frac(tick_ms):
    return 1.0 - (tick_ms - TICK_MIN_MS) / (TICK_MAX_MS - TICK_MIN_MS)


def frac_to_tick(frac):
    frac = max(0.0, min(1.0, frac))
    return int(round(TICK_MAX_MS - frac * (TICK_MAX_MS - TICK_MIN_MS)))


def draw_speed_slider(screen, rect, tick_ms, font):
    label = f"Speed ({tick_ms}ms/move) - drag, or +/-"
    text = font.render(label, True, HINT_COLOR)
    screen.blit(text, (rect.x, rect.y - 20))
    pygame.draw.rect(screen, (55, 58, 50), rect, border_radius=rect.height // 2)
    frac = tick_to_frac(tick_ms)
    handle_x = rect.x + frac * rect.width
    fill_rect = pygame.Rect(rect.x, rect.y, max(0, handle_x - rect.x), rect.height)
    pygame.draw.rect(screen, (90, 160, 90), fill_rect, border_radius=rect.height // 2)
    pygame.draw.circle(screen, (230, 232, 224), (int(handle_x), rect.y + rect.height // 2), SLIDER_HANDLE_R)


def slider_hit(rect, pos):
    """True if `pos` is close enough to the slider bar (a bit of vertical
    slack beyond `rect`, since the draggable handle sticks out above and
    below a thin bar) to count as a click/drag on it."""
    x, y = pos
    return rect.x - SLIDER_HANDLE_R <= x <= rect.x + rect.width + SLIDER_HANDLE_R \
        and rect.y - SLIDER_HANDLE_R <= y <= rect.y + rect.height + SLIDER_HANDLE_R


def tick_from_mouse_x(rect, mouse_x):
    frac = (mouse_x - rect.x) / rect.width
    return frac_to_tick(frac)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def cell_px(cell):
    x, y = cell
    return x * CELL, y * CELL


def draw_grid(screen):
    screen.fill(BG)
    for x in range(GRID_W + 1):
        pygame.draw.line(screen, GRID_LINE, (x * CELL, 0), (x * CELL, BOARD_H))
    for y in range(GRID_H + 1):
        pygame.draw.line(screen, GRID_LINE, (0, y * CELL), (BOARD_W, y * CELL))


def draw_food(screen, food, pulse_t):
    if food is None:
        return
    cx = food[0] * CELL + CELL / 2
    cy = food[1] * CELL + CELL / 2
    pulse = 0.5 + 0.5 * math.sin(pulse_t * 2 * math.pi)
    glow_r = CELL * (0.55 + 0.12 * pulse)
    glow_surf = pygame.Surface((int(glow_r * 2), int(glow_r * 2)), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*FOOD_GLOW, 70), (glow_r, glow_r), glow_r)
    screen.blit(glow_surf, (cx - glow_r, cy - glow_r))
    r = CELL * 0.30
    pygame.draw.circle(screen, FOOD_COLOR, (cx, cy), r)
    pygame.draw.circle(screen, (255, 210, 210), (cx - r * 0.3, cy - r * 0.3), r * 0.28)


def segment_from_positions(new_body, prev_body):
    """For each segment index i in new_body, the cell it's animating FROM
    this tick. new_body[i] always came from prev_body[i-1] (or, for the
    head, prev_body[0]) — a simple 'shift by one' mapping that's correct
    whether or not the snake grew this tick (see engine.step_snake's
    docstring for why: growth only ever changes whether the tail is
    dropped, never how the head/body segments shift)."""
    if not prev_body:
        return list(new_body)
    froms = []
    for i in range(len(new_body)):
        froms.append(prev_body[0] if i == 0 else prev_body[min(i - 1, len(prev_body) - 1)])
    return froms


def draw_snake(screen, body, prev_body, palette, t, alive=True, death_t=0.0):
    head_color, tail_color = palette
    froms = segment_from_positions(body, prev_body)
    n = max(len(body) - 1, 1)
    shrink = 1.0
    if not alive:
        shrink = max(0.0, 1.0 - death_t)
    for i, (cell, frm) in enumerate(zip(body, froms)):
        fx, fy = cell_px(frm)
        tx, ty = cell_px(cell)
        px = lerp(fx, tx, t if alive else 0.0)
        py = lerp(fy, ty, t if alive else 0.0)
        frac = i / n
        color = tuple(int(lerp(head_color[c], tail_color[c], frac)) for c in range(3))
        if not alive:
            color = tuple(int(lerp(color[c], DEATH_FLASH[c], 0.5)) for c in range(3))
        size = CELL * (0.86 if i > 0 else 0.92) * shrink
        offset = (CELL - size) / 2
        rect = pygame.Rect(px + offset, py + offset, size, size)
        pygame.draw.rect(screen, color, rect, border_radius=int(size * 0.35))
        if i == 0 and alive:
            _draw_eyes(screen, px, py, body)


def _draw_eyes(screen, px, py, body):
    heading = "up"
    if len(body) > 1:
        hx, hy = body[0]
        nx, ny = body[1]
        dx, dy = hx - nx, hy - ny
        for d, dd in DELTA.items():
            if dd == (dx, dy):
                heading = d
    ex, ey = {
        "up": (0.0, -0.12), "down": (0.0, 0.12),
        "left": (-0.16, -0.02), "right": (0.16, -0.02),
    }[heading]
    cx, cy = px + CELL / 2, py + CELL / 2
    perp_x, perp_y = (0.16 * CELL, 0) if heading in ("up", "down") else (0, 0.16 * CELL)
    for sign in (-1, 1):
        eye_x = cx + ex * CELL + perp_x * sign
        eye_y = cy + ey * CELL + perp_y * sign
        pygame.draw.circle(screen, (20, 20, 20), (eye_x, eye_y), CELL * 0.07)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color")

    def __init__(self, x, y, vx, vy, life, color):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color = color

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0

    def draw(self, screen):
        t = max(self.life / self.max_life, 0)
        r = 4 * t + 1
        alpha = int(255 * t)
        surf = pygame.Surface((int(r * 2), int(r * 2)), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), (r, r), r)
        screen.blit(surf, (self.x - r, self.y - r))


def spawn_burst(particles, cell, color, rng, count=14):
    cx, cy = cell[0] * CELL + CELL / 2, cell[1] * CELL + CELL / 2
    for _ in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(40, 140)
        particles.append(Particle(cx, cy, math.cos(angle) * speed, math.sin(angle) * speed,
                                   life=rng.uniform(0.25, 0.5), color=color))


# ---------------------------------------------------------------------------
# Best-score persistence (one running best across modes, like 2048's).
# ---------------------------------------------------------------------------

def load_best_score(repo_root):
    import json
    path = repo_root / "cache" / "apps" / "games" / "snake" / BEST_SCORE_FILENAME
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("best", 0))
    except (ValueError, OSError):
        return 0


def save_best_score(best, repo_root):
    import json
    path = repo_root / "cache" / "apps" / "games" / "snake" / BEST_SCORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best": int(best)}))


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class Player:
    """One snake on the board, human- or AI-controlled."""

    def __init__(self, snake_state, palette, controller):
        self.state = snake_state
        self.prev_body = list(snake_state.body)
        self.palette = palette
        self.controller = controller  # "human" or "ai"
        self.pending_direction = snake_state.direction
        self.death_t = 0.0
        # Every (resulting body, food) an AI-controlled snake has already
        # produced this game — see resolve_ai_direction below for why an
        # exact repeat here means a guaranteed infinite loop, not just bad
        # luck. Human play doesn't need this (a human breaks their own
        # loops); reset automatically every new_game since Player objects
        # aren't reused across games.
        self.seen_keys = set()

    def other_blocked(self, players):
        blocked = set()
        for p in players:
            if p is not self and p.state.alive:
                blocked |= p.state.occupied()
        return blocked


def new_game(mode, rng, num_arena_snakes=3):
    """Returns (players, food)."""
    players = []
    if mode in ("human", "ai"):
        snake = engine.new_snake(rng)
        controller = "human" if mode == "human" else "ai"
        players.append(Player(snake, SNAKE_PALETTE[0], controller))
    elif mode == "arena":
        starts = [
            (GRID_W // 4, GRID_H // 4), (3 * GRID_W // 4, GRID_H // 4),
            (GRID_W // 4, 3 * GRID_H // 4), (3 * GRID_W // 4, 3 * GRID_H // 4),
        ]
        dirs = ["right", "left", "right", "left"]
        for i in range(min(num_arena_snakes, len(starts))):
            snake = engine.new_snake(rng, start=starts[i], direction=dirs[i])
            players.append(Player(snake, SNAKE_PALETTE[i % len(SNAKE_PALETTE)], "ai"))
    occupied = set()
    for p in players:
        occupied |= p.state.occupied()
    food = engine.place_food(rng, occupied)
    return players, food


def resulting_body(body, direction, food):
    """What engine.step_snake would turn `body` into for this `direction`
    — computed without actually stepping, so candidate moves can be
    compared before committing to one. Mirrors step_snake's growth rule
    exactly (see its docstring): drop the tail unless this move eats."""
    dx, dy = DELTA[direction]
    head = body[0]
    new_head = (head[0] + dx, head[1] + dy)
    ate = food is not None and new_head == food
    new_body = [new_head] + body if ate else [new_head] + body[:-1]
    return tuple(new_body)


def resolve_ai_direction(p, food, other_blocked, heuristic_only, rng):
    """Picks this AI-controlled player's direction for the tick, with a
    cycle breaker layered on top of choose_ai_move/heuristic_choose_move.

    WHY THIS IS NEEDED: both the trained network and the heuristic solver
    are pure, deterministic functions of the current (body, direction,
    food, other_blocked) — there's no memory of past moves, no
    randomness. Snake's own dynamics are equally deterministic. So if an
    AI-controlled snake's (resulting body, food) ever exactly repeats a
    combination it's already produced this game, it is MATHEMATICALLY
    GUARANTEED to repeat the same decision from there and loop forever —
    this isn't "the network is imperfect," it's an unavoidable property
    of any deterministic reactive policy in a deterministic environment,
    and it can happen to a perfectly-trained network just as easily as an
    undertrained one.

    The fix: track every (resulting body, food) this player has already
    produced (Player.seen_keys). The network/heuristic's preferred move
    is tried first as always; if it would reproduce a seen state, the
    next-best candidate (ranked by non_trapping_moves' own safety
    ordering, then any remaining legal move) is tried instead, and so on,
    until one produces a genuinely new state. Only in the (extremely
    rare) case where EVERY available move would repeat a seen state is a
    move chosen at random — a last-resort tie-breaker, not the normal
    path, since at that point every deterministic choice is provably a
    dead end anyway.
    """
    primary = (agent.heuristic_choose_move(p.state.body, p.state.direction, food, other_blocked)
               if heuristic_only else
               agent.choose_ai_move(p.state.body, p.state.direction, food, other_blocked))
    if primary is None:
        return None  # no safe move at all; step_snake will resolve the (fatal) collision

    legal = engine.legal_directions(p.state.direction)
    head = p.state.head()
    own_blocked = set(p.state.body[:-1]) if len(p.state.body) > 1 else set()
    blocked = own_blocked | other_blocked
    safe = [d for d in legal
            if engine.in_bounds((head[0] + DELTA[d][0], head[1] + DELTA[d][1]), GRID_W, GRID_H)
            and (head[0] + DELTA[d][0], head[1] + DELTA[d][1]) not in blocked]
    pool = agent.non_trapping_moves(p.state.body, safe, other_blocked) or safe

    candidates = [primary] + [d for d in pool if d != primary]
    for d in candidates:
        key = (resulting_body(p.state.body, d, food), food)
        if key not in p.seen_keys:
            p.seen_keys.add(key)
            return d

    # Every candidate reproduces a state we've already been in — a
    # genuine forced loop (typically only one safe move existed anyway).
    # Random tie-break to escape rather than freezing on the same choice.
    d = rng.choice(candidates)
    p.seen_keys.add((resulting_body(p.state.body, d, food), food))
    return d


def step_all(players, food, rng, particles, heuristic_only=False):
    """Advances every alive player by one tick. Returns the (possibly
    new) food position."""
    for p in players:
        if not p.state.alive:
            continue
        other_blocked = p.other_blocked(players)
        if p.controller == "human":
            direction = p.pending_direction
            legal = engine.legal_directions(p.state.direction)
            if direction not in legal:
                direction = p.state.direction
        else:
            direction = resolve_ai_direction(p, food, other_blocked, heuristic_only, rng)
            if direction is None:
                direction = p.state.direction  # will collide; let step_snake settle it
        p.prev_body = list(p.state.body)
        p.state = engine.step_snake(p.state, direction, food, other_blocked)
        if p.state.just_ate:
            spawn_burst(particles, p.state.head(), p.palette[0], rng)
            occupied = set()
            for q in players:
                if q.state.alive:
                    occupied |= q.state.occupied()
            food = engine.place_food(rng, occupied)
    return food


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("TensorLang: Snake")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    big_font = pygame.font.SysFont(None, 40)

    repo_root = agent.chunked_runner.find_repo_root()
    best_score = load_best_score(repo_root)
    rng = random.Random()

    state = "start"  # start | playing | over
    mode = "human"
    players, food = [], None
    particles = []
    tick_ms = DEFAULT_TICK_MS
    tick_elapsed = 0.0
    heuristic_only = False
    winner_text = ""
    dragging_slider = False
    game_time = 0.0  # seconds elapsed in the current game; frozen once state == "over"

    start_slider_rect = pygame.Rect((WINDOW_W - 260) // 2, 340, 260, 10)
    play_slider_rect = pygame.Rect(16, BOARD_H + 78, WINDOW_W - 32, 8)

    def start(new_mode):
        nonlocal players, food, state, mode, tick_elapsed, game_time
        mode = new_mode
        players, food = new_game(mode, rng)
        tick_elapsed = 0.0
        game_time = 0.0
        state = "playing"

    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    while True:
        dt = clock.tick(60) / 1000.0
        active_slider_rect = start_slider_rect if state == "start" else play_slider_rect
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if slider_hit(active_slider_rect, event.pos):
                    dragging_slider = True
                    tick_ms = tick_from_mouse_x(active_slider_rect, event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_slider = False
            elif event.type == pygame.MOUSEMOTION and dragging_slider:
                tick_ms = tick_from_mouse_x(active_slider_rect, event.pos[0])
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if event.key == pygame.K_r:
                    state = "start"
                if state == "start":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE) or event.key in ARROW_KEYS:
                        start("human")
                    elif event.key == pygame.K_a:
                        start("ai")
                    elif event.key == pygame.K_m:
                        start("arena")
                elif state in ("playing", "over"):
                    if event.key == pygame.K_n:
                        start(mode)
                    if event.key == pygame.K_h and mode in ("ai", "arena"):
                        heuristic_only = not heuristic_only
                    if state == "playing" and mode == "human" and event.key in KEY_TO_DIRECTION:
                        players[0].pending_direction = KEY_TO_DIRECTION[event.key]
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        tick_ms = max(TICK_MIN_MS, tick_ms - TICK_STEP_MS)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        tick_ms = min(TICK_MAX_MS, tick_ms + TICK_STEP_MS)

        if state == "start":
            screen.fill(BG)
            title = big_font.render("Snake on TensorLang", True, TEXT_COLOR)
            screen.blit(title, ((WINDOW_W - title.get_width()) // 2, 90))
            lines = [
                "Enter / arrows: play it yourself",
                "A: watch one AI snake play",
                "M: arena - several AI snakes compete",
                f"Best score: {best_score}",
            ]
            y = 190
            for line in lines:
                surf = font.render(line, True, HINT_COLOR)
                screen.blit(surf, ((WINDOW_W - surf.get_width()) // 2, y))
                y += 32
            draw_speed_slider(screen, start_slider_rect, tick_ms, small_font)
            pygame.display.flip()
            continue

        if state == "playing":
            game_time += dt
            tick_elapsed += dt * 1000.0
            if tick_elapsed >= tick_ms:
                tick_elapsed -= tick_ms
                food = step_all(players, food, rng, particles, heuristic_only)
                for p in players:
                    if p.state.score > best_score:
                        best_score = p.state.score
                        save_best_score(best_score, repo_root)
                alive = [p for p in players if p.state.alive]
                if mode in ("human", "ai") and not alive:
                    state = "over"
                    winner_text = f"Game over - score {players[0].state.score}"
                elif mode == "arena" and len(alive) <= (1 if len(players) > 1 else 0):
                    state = "over"
                    if len(alive) == 1:
                        idx = players.index(alive[0])
                        winner_text = f"Snake {idx + 1} wins! Score {alive[0].state.score}"
                    else:
                        winner_text = "Everyone crashed - draw!"

        t = min(tick_elapsed / tick_ms, 1.0)
        draw_grid(screen)
        draw_food(screen, food, time.time() % 1.0)
        for p in players:
            draw_snake(screen, p.state.body, p.prev_body, p.palette, t,
                       alive=p.state.alive, death_t=p.death_t)
            if not p.state.alive:
                p.death_t = min(1.0, p.death_t + dt * 2.0)

        particles[:] = [pt for pt in particles if pt.update(dt)]
        for pt in particles:
            pt.draw(screen)

        pygame.draw.rect(screen, BG, (0, BOARD_H, WINDOW_W, STATUS_H))
        driver_label = "heuristic" if heuristic_only else "network"
        time_label = format_time(game_time)
        if mode == "arena":
            scores = "  ".join(f"S{i+1}:{p.state.score}" for i, p in enumerate(players))
            line1 = f"Arena - {scores}    Best: {best_score}    Time: {time_label}"
            line2 = f"H: {driver_label}   N: new game   R: menu"
        elif mode == "ai":
            line1 = (f"AI - Score: {players[0].state.score if players else 0}    "
                     f"Best: {best_score}    Time: {time_label}")
            line2 = f"H: {driver_label}   N: new game   R: menu"
        else:
            line1 = (f"You - Score: {players[0].state.score if players else 0}    "
                     f"Best: {best_score}    Time: {time_label}")
            line2 = "N: new game   R: menu"
        screen.blit(font.render(line1, True, TEXT_COLOR), (12, BOARD_H + 10))
        screen.blit(small_font.render(line2, True, HINT_COLOR), (12, BOARD_H + 34))
        draw_speed_slider(screen, play_slider_rect, tick_ms, small_font)
        if state == "over":
            screen.blit(font.render(winner_text + "   (N: new game, R: menu)", True, (255, 200, 120)),
                        (12, BOARD_H + 100))

        pygame.display.flip()


if __name__ == "__main__":
    main()

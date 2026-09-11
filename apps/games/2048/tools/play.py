#!/usr/bin/env python3
"""
Interactive 2048, played on the real TensorLang tensor-ops engine — or sit
back and watch the trained AI play it.

Run from the tensor-lang repo root:

    python3 apps/games/2048/tools/play.py

Controls:
    Arrow keys / WASD   slide the board (human mode)
    B                   watch the AI play, back to back
    N                   new game, any time
    + / -               during autoplay: speed up / down
    R                   back to the start screen
    ESC or Q            quit

Every move — whichever mode picked the direction — is actually computed
by running the matching step_*.tl file (step.tl / step_right.tl /
step_up.tl / step_down.tl) as a TensorLang subprocess; see tools/agent.py
for that wiring and its fallback if the engine call fails. The very first
run of each direction pays a one-time CUDA kernel compile (which can take
tens of seconds); this file pre-compiles all four with a "warming up"
screen before you can start playing, specifically so that delay doesn't
happen mid-game. Every engine call — warm-up included — polls the
subprocess instead of blocking on it, so the window keeps pumping events
and redrawing the whole time; without that, the OS's "app not responding"
watchdog fires during any wait longer than a few seconds, which is
exactly what a blocking call during a 30+s first-time compile looks like.

Autoplay uses tools/agent.py's choose_ai_move — the trained TensorLang
policy network (infer.tl), the same way tic_tac_toe's choose_move calls
its infer.tl. If the trained weights aren't there yet (run
`./run.sh --train` first) or inference fails for any reason, choose_ai_move
falls back to the original hand-written heuristic instead, printing a
warning. Only the DECISION of which way to slide comes from either of
those; the resulting board is still always computed by the real engine.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

BOARD_N = 4
WINDOW_SIZE = 500
MARGIN = 16
LINE_WIDTH = 0
GRID = WINDOW_SIZE - 2 * MARGIN
CELL = GRID // BOARD_N
CELL_PAD = 8
STATUS_H = 110

BG = (250, 248, 239)
GRID_BG = (187, 173, 160)
EMPTY_CELL = (205, 193, 180)
TEXT_COLOR = (60, 56, 50)
HINT_COLOR = (120, 116, 108)

TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
TILE_COLOR_DEFAULT = (60, 58, 50)  # anything past 2048
DARK_TEXT = (119, 110, 101)
LIGHT_TEXT = (249, 246, 242)

# Autoplay pacing, same idea as tic_tac_toe's SELF_PLAY_MOVE_DELAY_MS.
AUTOPLAY_MOVE_DELAY_MS = 200
AUTOPLAY_DELAY_MIN_MS = 0
AUTOPLAY_DELAY_MAX_MS = 2000
AUTOPLAY_DELAY_STEP_MS = 100

KEY_TO_DIRECTION = {
    pygame.K_LEFT: "left", pygame.K_a: "left",
    pygame.K_RIGHT: "right", pygame.K_d: "right",
    pygame.K_UP: "up", pygame.K_w: "up",
    pygame.K_DOWN: "down", pygame.K_s: "down",
}


def cell_rect(i):
    row, col = divmod(i, BOARD_N)
    return pygame.Rect(MARGIN + col * CELL, MARGIN + row * CELL, CELL, CELL)


def tile_color(value):
    return TILE_COLORS.get(value, TILE_COLOR_DEFAULT)


def tile_text_color(value):
    return DARK_TEXT if value <= 4 else LIGHT_TEXT


def font_for_value(value, big_font, med_font, small_font):
    digits = len(str(value))
    if digits <= 2:
        return big_font
    if digits == 3:
        return med_font
    return small_font


def draw_board(screen, board, fonts):
    big_font, med_font, small_font = fonts
    pygame.draw.rect(screen, GRID_BG, (0, 0, WINDOW_SIZE, WINDOW_SIZE), border_radius=10)
    for i, value in enumerate(board):
        rect = cell_rect(i).inflate(-CELL_PAD, -CELL_PAD)
        pygame.draw.rect(screen, tile_color(value), rect, border_radius=6)
        if value:
            font = font_for_value(value, big_font, med_font, small_font)
            surf = font.render(str(value), True, tile_text_color(value))
            screen.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2))


def draw_status(screen, font, small_font, lines):
    if isinstance(lines, str):
        lines = [lines]
    pygame.draw.rect(screen, BG, (0, WINDOW_SIZE, WINDOW_SIZE, STATUS_H))
    line_h = font.get_height() + 4
    y = WINDOW_SIZE + 10
    for i, text in enumerate(lines):
        f = font if i == 0 else small_font
        surf = f.render(text, True, TEXT_COLOR if i == 0 else HINT_COLOR)
        screen.blit(surf, (MARGIN, y))
        y += line_h


def draw_start_screen(screen, font, big_font, best_score):
    screen.fill(BG)
    title = big_font.render("2048 on TensorLang", True, TEXT_COLOR)
    screen.blit(title, ((WINDOW_SIZE - title.get_width()) // 2, 90))
    prompts = [
        "Arrow keys / WASD to play",
        "Press B to watch the AI play",
        f"Best score: {best_score}",
    ]
    y = 190
    for p in prompts:
        surf = font.render(p, True, HINT_COLOR)
        screen.blit(surf, ((WINDOW_SIZE - surf.get_width()) // 2, y))
        y += 32
    pygame.display.flip()


def fmt_ms(ms):
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def draw_engine_wait(screen, fonts, board, label, elapsed):
    """Overlay shown while a step_*.tl subprocess is running. Drawn once
    per poll tick (see run_engine_polling/warm_up_engine below) so the
    window keeps redrawing and pumping events instead of looking frozen."""
    draw_board(screen, board, fonts)
    box = pygame.Rect(0, WINDOW_SIZE // 2 - 55, WINDOW_SIZE, 110)
    overlay = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
    overlay.fill((20, 18, 16, 210))
    screen.blit(overlay, box.topleft)
    big, _, _ = fonts
    small_font = pygame.font.SysFont(None, 20)
    title = pygame.font.SysFont(None, 26).render(label, True, LIGHT_TEXT)
    screen.blit(title, ((WINDOW_SIZE - title.get_width()) // 2, box.top + 22))
    note = f"{elapsed:.1f}s elapsed"
    if elapsed > 3:
        note += " — first-time GPU kernel compiles can take up to ~40s"
    sub = small_font.render(note, True, (210, 205, 198))
    screen.blit(sub, ((WINDOW_SIZE - sub.get_width()) // 2, box.top + 58))


def _poll_subprocess(screen, fonts, board, label, proc):
    """Shared poll loop: pumps events (handling QUIT so the window can
    still be closed mid-compile) and redraws draw_engine_wait every tick
    until `proc` finishes. Returns elapsed seconds."""
    t0 = time.perf_counter()
    while proc.poll() is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                proc.kill()
                pygame.quit()
                sys.exit(0)
        elapsed = time.perf_counter() - t0
        draw_engine_wait(screen, fonts, board, label, elapsed)
        pygame.display.flip()
        time.sleep(0.05)
    return time.perf_counter() - t0


def run_engine_polling(screen, fonts, board, direction, repo_root):
    """Like agent.apply_move, but polls the step_*.tl subprocess instead
    of blocking on it (see agent.spawn_move_engine/collect_move_engine),
    redrawing a wait overlay each tick so a slow first-time kernel compile
    never makes the window look unresponsive. Returns the same tuple
    do_move used to: (new_board, moved, gained, used_engine, elapsed_ms)."""
    sim_board, moved, gained = agent.simulate_move(board, direction)
    if not moved:
        return board, False, 0, False, 0.0

    proc, step_dir = agent.spawn_move_engine(board, direction, repo_root)
    elapsed_s = _poll_subprocess(screen, fonts, board, f"Sliding {direction}...", proc)
    elapsed_ms = elapsed_s * 1000

    try:
        engine_board = agent.collect_move_engine(proc, step_dir, agent.STEP_FILE[direction])
    except Exception as e:  # noqa: BLE001 - same broad catch as agent.apply_move, see its docstring
        print(f"[play] {agent.STEP_FILE[direction]} failed, falling back to the plain-Python move: {e}")
        return sim_board, True, gained, False, elapsed_ms

    if engine_board != sim_board:
        print(
            f"[play] WARNING: engine result disagrees with the plain-Python reference "
            f"for direction={direction!r}\n  input : {board}\n  engine: {engine_board}\n  python: {sim_board}"
        )
    return engine_board, True, gained, True, elapsed_ms


def warm_up_engine(screen, fonts, repo_root):
    """Pre-compiles all four directions' CUDA kernels once, up front,
    before the player can start moving — so the ~30-40s first-time
    compile (see NOTES.md) happens here, with an honest progress screen,
    instead of ambushing the first real move or the first autoplay move.
    Already-cached kernels make this finish in a couple of seconds total,
    so this runs unconditionally every launch rather than trying to detect
    whether it's needed."""
    warm_board = [2] + [0] * 15
    for i, direction in enumerate(agent.DIRECTIONS):
        label = f"Warming up the TensorLang engine ({i + 1}/{len(agent.DIRECTIONS)}): {direction}"
        proc, step_dir = agent.spawn_move_engine(warm_board, direction, repo_root)
        _poll_subprocess(screen, fonts, warm_board, label, proc)
        try:
            agent.collect_move_engine(proc, step_dir, agent.STEP_FILE[direction])
        except Exception as e:  # noqa: BLE001
            print(f"[play] warm-up for {direction!r} failed (will fall back to Python during play): {e}")


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + STATUS_H))
    pygame.display.set_caption("TensorLang: 2048")
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    big_font_title = pygame.font.SysFont(None, 34)
    tile_big = pygame.font.SysFont(None, 46, bold=True)
    tile_med = pygame.font.SysFont(None, 36, bold=True)
    tile_small = pygame.font.SysFont(None, 26, bold=True)
    fonts = (tile_big, tile_med, tile_small)
    clock = pygame.time.Clock()
    rng = np.random.default_rng()

    repo_root = agent.chunked_runner.find_repo_root()
    best_score = agent.load_best_score(repo_root)

    warm_up_engine(screen, fonts, repo_root)

    # state: "start" | "playing" | "over" | "won" | "autoplaying"
    state = "start"
    board = [0] * (BOARD_N * BOARD_N)
    score = 0
    last_move_ms = None
    last_used_engine = None
    autoplay_delay_ms = AUTOPLAY_MOVE_DELAY_MS
    next_ai_at = 0
    won_continue = False  # once you dismiss the "you won" banner, keep playing

    def start_game():
        nonlocal board, score, state, last_move_ms, last_used_engine, won_continue
        board = agent.new_board(rng)
        score = 0
        last_move_ms = None
        last_used_engine = None
        won_continue = False
        state = "playing"

    def start_autoplay():
        nonlocal board, score, state, last_move_ms, last_used_engine, next_ai_at, won_continue
        board = agent.new_board(rng)
        score = 0
        last_move_ms = None
        last_used_engine = None
        won_continue = False
        state = "autoplaying"
        next_ai_at = pygame.time.get_ticks()

    def finish_move(new_board, moved, gained):
        nonlocal board, score, state, best_score
        if not moved:
            return
        board = new_board
        score += gained
        if score > best_score:
            best_score = score
            agent.save_best_score(best_score, repo_root)
        agent.spawn_tile(board, rng)
        if agent.has_won(board) and not won_continue:
            state = "won"
        elif agent.is_game_over(board):
            state = "over"

    while True:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if event.key == pygame.K_r:
                    state = "start"
                if event.key == pygame.K_n and state != "start":
                    start_game()
                if state == "start":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE) or event.key in KEY_TO_DIRECTION:
                        start_game()
                    elif event.key == pygame.K_b:
                        start_autoplay()
                if state == "won" and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    won_continue = True
                    state = "playing"
                if state == "playing" and event.key in KEY_TO_DIRECTION:
                    direction = KEY_TO_DIRECTION[event.key]
                    new_board, moved, gained, used_engine, elapsed_ms = run_engine_polling(screen, fonts, board, direction, repo_root)
                    last_move_ms = elapsed_ms
                    last_used_engine = used_engine
                    finish_move(new_board, moved, gained)
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    autoplay_delay_ms = max(AUTOPLAY_DELAY_MIN_MS, autoplay_delay_ms - AUTOPLAY_DELAY_STEP_MS)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    autoplay_delay_ms = min(AUTOPLAY_DELAY_MAX_MS, autoplay_delay_ms + AUTOPLAY_DELAY_STEP_MS)

        if state == "start":
            draw_start_screen(screen, font, big_font_title, best_score)
            clock.tick(30)
            continue

        if state == "autoplaying" and now >= next_ai_at:
            direction = agent.choose_ai_move(board)
            if direction is None:
                state = "over"
            else:
                new_board, moved, gained, used_engine, elapsed_ms = run_engine_polling(screen, fonts, board, direction, repo_root)
                last_move_ms = elapsed_ms
                last_used_engine = used_engine
                finish_move(new_board, moved, gained)
                next_ai_at = now + autoplay_delay_ms
                if state == "won":
                    won_continue = True
                    state = "autoplaying"

        screen.fill(BG)
        draw_board(screen, board, fonts)

        timing = ""
        if last_move_ms is not None:
            engine_note = "engine" if last_used_engine else "python fallback"
            timing = f"   last move: {fmt_ms(last_move_ms)} ({engine_note})"

        if state == "playing":
            draw_status(screen, font, small_font, [
                f"Score: {score}    Best: {best_score}{timing}",
                "Arrow keys / WASD to move. N: new game   B: autoplay   R: menu",
            ])
        elif state == "autoplaying":
            draw_status(screen, font, small_font, [
                f"Autoplay — Score: {score}    Best: {best_score}{timing}",
                f"Speed: {autoplay_delay_ms}ms (+/-)   N: new game   R: menu",
            ])
        elif state == "won":
            draw_status(screen, font, small_font, [
                f"You reached 2048! Score: {score}    Best: {best_score}",
                "Enter/Space to keep playing   N: new game   R: menu",
            ])
        elif state == "over":
            draw_status(screen, font, small_font, [
                f"Game over — Score: {score}    Best: {best_score}",
                "N: new game   R: menu",
            ])

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()

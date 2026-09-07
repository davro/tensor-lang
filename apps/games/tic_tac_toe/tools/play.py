#!/usr/bin/env python3
"""
Interactive tic-tac-toe: you vs. the TensorLang-trained policy network —
or sit back and watch TensorLang play itself.

Run from the tensor-lang repo root (after run.sh has trained the network
at least once):

    python3 apps/games/tic_tac_toe/tools/play.py

Controls:
    X / O            play a match yourself, as X or O
    S                watch TensorLang play itself, back to back
    Click a cell     your move, on your turn (human modes only)
    + / -            during self-play: speed matches up / down
    R                back to the start screen, any time
    ESC or Q         quit

Each of the AI's moves runs apps/games/tic_tac_toe/infer.tl as a
TensorLang subprocess. The first move in a session pays a one-time CUDA
kernel compile; every move after that reuses the cached kernel (see the
compiler's kernel-cache-skip patch), so moves after the first are fast —
watch the "last move" timing readout during a match to see this live.

Self-play SAMPLES from the network's move distribution instead of always
taking its single best move — the network itself is deterministic, so
"best vs best" would replay the exact same game forever, which isn't
much of a show. Sampling means every self-play match plays out a little
differently while still following what the network actually learned.
Its scoreboard persists across sessions (cache/apps/games/tic_tac_toe/
self_play_stats.json), so closing the window doesn't lose the tally.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

WINDOW_SIZE = 480
MARGIN = 30
LINE_WIDTH = 6
GRID = WINDOW_SIZE - 2 * MARGIN
CELL = GRID // 3
STATUS_H = 90

BG = (245, 241, 230)
LINE_COLOR = (60, 56, 50)
X_COLOR = (210, 60, 60)
O_COLOR = (50, 90, 200)
TEXT_COLOR = (40, 38, 34)
HINT_COLOR = (120, 116, 108)
WIN_LINE_COLOR = (210, 160, 40)
CONF_COLOR = (150, 146, 136)

HUMAN, AI = "human", "ai"

WINNING_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

# Self-play pacing: the network itself may answer near-instantly once its
# kernel is cached, so this keeps matches watchable rather than flickering
# past in a fraction of a second. Adjustable in-game with +/-.
SELF_PLAY_MOVE_DELAY_MS = 450
SELF_PLAY_DELAY_MIN_MS = 0
SELF_PLAY_DELAY_MAX_MS = 3000
SELF_PLAY_DELAY_STEP_MS = 150
SELF_PLAY_RESULT_PAUSE_MS = 2200


def cell_rect(i):
    row, col = divmod(i, 3)
    return pygame.Rect(MARGIN + col * CELL, MARGIN + row * CELL, CELL, CELL)


def cell_at_pos(pos):
    x, y = pos
    if not (MARGIN <= x < MARGIN + GRID and MARGIN <= y < MARGIN + GRID):
        return None
    col = (x - MARGIN) // CELL
    row = (y - MARGIN) // CELL
    return int(row * 3 + col)


def winning_line(board):
    """Returns the (a, b, c) cell-index triple of a completed line, or
    None if there isn't one (including "no winner yet")."""
    for a, b, c in WINNING_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return (a, b, c)
    return None


def winner(board):
    line = winning_line(board)
    if line is not None:
        return board[line[0]]
    if all(v != 0 for v in board):
        return 0  # draw
    return None  # game still in progress


def draw_grid(screen):
    for i in (1, 2):
        pygame.draw.line(screen, LINE_COLOR, (MARGIN + i * CELL, MARGIN), (MARGIN + i * CELL, MARGIN + GRID), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (MARGIN, MARGIN + i * CELL), (MARGIN + GRID, MARGIN + i * CELL), LINE_WIDTH)


def draw_mark(screen, i, value):
    rect = cell_rect(i)
    pad = CELL * 0.22
    if value == 1:  # X
        pygame.draw.line(screen, X_COLOR, (rect.left + pad, rect.top + pad), (rect.right - pad, rect.bottom - pad), 10)
        pygame.draw.line(screen, X_COLOR, (rect.right - pad, rect.top + pad), (rect.left + pad, rect.bottom - pad), 10)
    elif value == -1:  # O
        pygame.draw.circle(screen, O_COLOR, rect.center, CELL // 2 - pad, 8)


def draw_winning_line(screen, board):
    line = winning_line(board)
    if line is None:
        return
    a, c = line[0], line[2]
    start, end = cell_rect(a).center, cell_rect(c).center
    pygame.draw.line(screen, WIN_LINE_COLOR, start, end, 10)


def draw_confidence(screen, small_font, board, probs):
    """Overlays the network's own move-probability (as a percentage) on
    every empty cell — a little window into what the "TensorLang player"
    is actually weighing, since that's the whole point of this app.

    `probs` is the network's raw, UNMASKED softmax output over all 9
    cells — it can (and does) put real probability mass on cells that
    are already occupied, since nothing in training penalizes that
    (choose_move separately masks + renormalizes before actually picking
    a move). Displaying that raw distribution directly meant the empty
    cells often showed as "0%" even when one was clearly the network's
    preferred legal move, because most of its confidence was sitting on
    an occupied cell we don't even draw a label for. Renormalizing over
    just the legal cells here makes the displayed percentages sum to
    100% among what you can actually watch it choose between.
    """
    if probs is None:
        return
    legal = [i for i, v in enumerate(board) if v == 0]
    if not legal:
        return
    legal_mass = max(float(probs[legal].sum()), 1e-8)
    for i in legal:
        pct = f"{probs[i] / legal_mass * 100:.0f}%"
        surf = small_font.render(pct, True, CONF_COLOR)
        rect = cell_rect(i)
        screen.blit(surf, (rect.centerx - surf.get_width() // 2, rect.bottom - surf.get_height() - 6))


def draw_status(screen, font, lines):
    """lines: a string, or a list of strings stacked top to bottom."""
    if isinstance(lines, str):
        lines = [lines]
    pygame.draw.rect(screen, BG, (0, WINDOW_SIZE, WINDOW_SIZE, STATUS_H))
    line_h = font.get_height() + 2
    total_h = line_h * len(lines)
    top = WINDOW_SIZE + (STATUS_H - total_h) // 2
    for i, text in enumerate(lines):
        surf = font.render(text, True, TEXT_COLOR)
        screen.blit(surf, (MARGIN, top + i * line_h))


def draw_start_screen(screen, font, big_font):
    screen.fill(BG)
    title = big_font.render("Tic-Tac-Toe vs TensorLang", True, TEXT_COLOR)
    screen.blit(title, ((WINDOW_SIZE - title.get_width()) // 2, 100))
    prompts = [
        "Press X to play as X (go first)",
        "Press O to play as O (go second)",
        "Press S to watch TensorLang play itself",
    ]
    y = 190
    for p in prompts:
        surf = font.render(p, True, HINT_COLOR)
        screen.blit(surf, ((WINDOW_SIZE - surf.get_width()) // 2, y))
        y += 32
    pygame.display.flip()


def fmt_ms(ms):
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + STATUS_H))
    pygame.display.set_caption("TensorLang: Tic-Tac-Toe")
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    big_font = pygame.font.SysFont(None, 34)
    clock = pygame.time.Clock()
    rng = np.random.default_rng()

    repo_root = agent.chunked_runner.find_repo_root()
    weights_dir = repo_root / "cache" / "apps" / "games" / "tic_tac_toe" / "train.tl" / "weights"
    if not (weights_dir / "w1.npy").exists():
        raise SystemExit(
            f"No trained weights found at {weights_dir}.\n"
            "Run ./apps/games/tic_tac_toe/run.sh first (it trains automatically)."
        )

    # state: "start" | "playing" | "over" | "self_playing" | "self_over"
    state = "start"
    board = [0] * 9
    human_mark = 1
    ai_mark = -1
    turn = HUMAN
    result_text = ""
    last_move_ms = None   # timing of the most recent AI inference call
    last_probs = None     # that move's raw probability array, for the overlay

    # Self-play only:
    self_turn = 1          # which mark moves next
    self_starter = 1       # who starts the *next* round (alternates each round)
    self_tally_raw = agent.load_self_play_stats(repo_root)
    self_tally = {1: self_tally_raw.get("1", 0), -1: self_tally_raw.get("-1", 0), 0: self_tally_raw.get("0", 0)}
    self_play_delay_ms = SELF_PLAY_MOVE_DELAY_MS
    next_ai_at = 0          # pygame.time timestamp gating the next AI move
    next_round_at = 0       # timestamp gating auto-restart after a self-play result

    def start_self_play():
        nonlocal board, self_turn, state, result_text, last_probs
        board = [0] * 9
        self_turn = self_starter
        result_text = ""
        last_probs = None
        state = "self_playing"

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
                if event.key == pygame.K_r and state != "start":
                    state = "start"
                if state == "start" and event.key in (pygame.K_x, pygame.K_o):
                    human_mark = 1 if event.key == pygame.K_x else -1
                    ai_mark = -human_mark
                    board = [0] * 9
                    result_text = ""
                    last_probs = None
                    turn = HUMAN if human_mark == 1 else AI
                    state = "playing"
                if state == "start" and event.key == pygame.K_s:
                    self_starter = 1
                    next_ai_at = now
                    start_self_play()
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self_play_delay_ms = max(SELF_PLAY_DELAY_MIN_MS, self_play_delay_ms - SELF_PLAY_DELAY_STEP_MS)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self_play_delay_ms = min(SELF_PLAY_DELAY_MAX_MS, self_play_delay_ms + SELF_PLAY_DELAY_STEP_MS)
            if event.type == pygame.MOUSEBUTTONDOWN and state == "playing" and turn == HUMAN:
                i = cell_at_pos(event.pos)
                if i is not None and board[i] == 0:
                    board[i] = human_mark
                    last_probs = None
                    w = winner(board)
                    if w is not None:
                        state = "over"
                        result_text = "You win!" if w == human_mark else ("Draw." if w == 0 else "TensorLang wins.")
                    else:
                        turn = AI

        if state == "start":
            draw_start_screen(screen, font, big_font)
            clock.tick(30)
            continue

        # Draw the current board up front, so the last move made (by
        # either side) is visible while any status text below updates.
        screen.fill(BG)
        draw_grid(screen)
        for i, v in enumerate(board):
            draw_mark(screen, i, v)
        if state in ("playing",) and turn == AI:
            draw_confidence(screen, small_font, board, last_probs)

        if state == "playing" and turn == AI:
            draw_status(screen, font, "TensorLang is thinking...")
            pygame.display.flip()
            t0 = time.perf_counter()
            move, probs = agent.choose_move(board, ai_mark, repo_root, mode="best")
            last_move_ms = (time.perf_counter() - t0) * 1000
            last_probs = probs
            board[move] = ai_mark
            w = winner(board)
            if w is not None:
                state = "over"
                result_text = "You win!" if w == human_mark else ("Draw." if w == 0 else "TensorLang wins.")
            else:
                turn = HUMAN
            screen.fill(BG)
            draw_grid(screen)
            for i, v in enumerate(board):
                draw_mark(screen, i, v)

        elif state == "self_playing" and now >= next_ai_at:
            mark_name = "X" if self_turn == 1 else "O"
            draw_confidence(screen, small_font, board, last_probs)
            draw_status(screen, font, [
                f"TensorLang ({mark_name}) is thinking...",
                f"X wins: {self_tally[1]}   O wins: {self_tally[-1]}   Draws: {self_tally[0]}",
            ])
            pygame.display.flip()
            t0 = time.perf_counter()
            move, probs = agent.choose_move(board, self_turn, repo_root, mode="sample", rng=rng)
            last_move_ms = (time.perf_counter() - t0) * 1000
            last_probs = probs
            board[move] = self_turn
            w = winner(board)
            screen.fill(BG)
            draw_grid(screen)
            for i, v in enumerate(board):
                draw_mark(screen, i, v)
            if w is not None:
                self_tally[w] += 1
                agent.save_self_play_stats(self_tally, repo_root)
                self_starter = -self_starter  # alternate who opens the next round
                result_text = "X wins!" if w == 1 else ("O wins!" if w == -1 else "Draw.")
                state = "self_over"
                next_round_at = now + SELF_PLAY_RESULT_PAUSE_MS
            else:
                self_turn = -self_turn
                next_ai_at = now + self_play_delay_ms

        elif state == "self_over" and now >= next_round_at:
            start_self_play()

        if state in ("over", "self_over"):
            draw_winning_line(screen, board)

        if state == "over":
            timing = f"   (last move: {fmt_ms(last_move_ms)})" if last_move_ms is not None else ""
            draw_status(screen, font, f"{result_text}   Press R to play again, Esc to quit.{timing}")
        elif state == "self_over":
            draw_status(screen, font, [
                f"{result_text}   Next match starting...",
                f"X wins: {self_tally[1]}   O wins: {self_tally[-1]}   Draws: {self_tally[0]}   (R for menu)",
            ])
        elif state == "self_playing":
            timing = f"   last move: {fmt_ms(last_move_ms)}" if last_move_ms is not None else ""
            draw_status(screen, font, [
                f"TensorLang self-play — {'X' if self_turn == 1 else 'O'} to move.{timing}",
                f"X wins: {self_tally[1]}   O wins: {self_tally[-1]}   Draws: {self_tally[0]}   "
                f"Speed: {self_play_delay_ms}ms (+/-)   (R for menu)",
            ])
        elif turn == HUMAN:
            draw_status(screen, font, "Your move — click a cell.")
        else:
            timing = f"   (last move: {fmt_ms(last_move_ms)})" if last_move_ms is not None else ""
            draw_status(screen, font, f"TensorLang is thinking...{timing}")

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()

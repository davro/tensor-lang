#!/usr/bin/env python3
"""
Interactive tic-tac-toe: you vs. the TensorLang-trained policy network.

Run from the tensor-lang repo root (after run.sh has trained the network
at least once):

    python3 apps/games/tic_tac_toe/tools/play.py

Controls:
    Click an empty cell to play there on your turn.
    X / O            choose your mark on the start screen
    R                restart (any time)
    ESC or Q         quit

Each of the AI's moves runs apps/games/tic_tac_toe/infer.tl as a fresh
TensorLang subprocess, which recompiles a small CUDA kernel via nvcc
every time (this compiler doesn't cache compiled kernels between runs —
see HANDOVER.md). That means the AI takes a couple of seconds to "think"
on every move; the window shows a status message while it works.
"""
import sys
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

WINDOW_SIZE = 480
MARGIN = 30
LINE_WIDTH = 6
GRID = WINDOW_SIZE - 2 * MARGIN
CELL = GRID // 3
STATUS_H = 60

BG = (245, 241, 230)
LINE_COLOR = (60, 56, 50)
X_COLOR = (210, 60, 60)
O_COLOR = (50, 90, 200)
TEXT_COLOR = (40, 38, 34)
HINT_COLOR = (120, 116, 108)

HUMAN, AI = "human", "ai"


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


def winner(board):
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for a, b, c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
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


def draw_status(screen, font, text):
    pygame.draw.rect(screen, BG, (0, WINDOW_SIZE, WINDOW_SIZE, STATUS_H))
    surf = font.render(text, True, TEXT_COLOR)
    screen.blit(surf, (MARGIN, WINDOW_SIZE + (STATUS_H - surf.get_height()) // 2))


def draw_start_screen(screen, font, big_font):
    screen.fill(BG)
    title = big_font.render("Tic-Tac-Toe vs TensorLang", True, TEXT_COLOR)
    screen.blit(title, ((WINDOW_SIZE - title.get_width()) // 2, 120))
    prompt1 = font.render("Press X to play as X (go first)", True, HINT_COLOR)
    prompt2 = font.render("Press O to play as O (go second)", True, HINT_COLOR)
    screen.blit(prompt1, ((WINDOW_SIZE - prompt1.get_width()) // 2, 220))
    screen.blit(prompt2, ((WINDOW_SIZE - prompt2.get_width()) // 2, 250))
    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + STATUS_H))
    pygame.display.set_caption("TensorLang: Tic-Tac-Toe")
    font = pygame.font.SysFont(None, 26)
    big_font = pygame.font.SysFont(None, 34)
    clock = pygame.time.Clock()

    repo_root = agent.chunked_runner.find_repo_root()
    weights_dir = repo_root / "cache" / "apps" / "games" / "tic_tac_toe" / "train.tl" / "weights"
    if not (weights_dir / "w1.npy").exists():
        raise SystemExit(
            f"No trained weights found at {weights_dir}.\n"
            "Run ./apps/games/tic_tac_toe/run.sh first (it trains automatically)."
        )

    state = "start"  # "start" | "playing" | "over"
    board = [0] * 9
    human_mark = 1
    ai_mark = -1
    turn = HUMAN
    result_text = ""

    while True:
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
                    turn = HUMAN if human_mark == 1 else AI
                    state = "playing"
            if event.type == pygame.MOUSEBUTTONDOWN and state == "playing" and turn == HUMAN:
                i = cell_at_pos(event.pos)
                if i is not None and board[i] == 0:
                    board[i] = human_mark
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

        # Draw the current board before a possibly-slow AI turn, so the
        # human's last move is visible while the status line updates.
        screen.fill(BG)
        draw_grid(screen)
        for i, v in enumerate(board):
            draw_mark(screen, i, v)

        if state == "playing" and turn == AI:
            draw_status(screen, font, "TensorLang is thinking... (compiling a CUDA kernel, a few seconds)")
            pygame.display.flip()
            move = agent.choose_move(board, ai_mark, repo_root)
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

        if state == "over":
            draw_status(screen, font, f"{result_text}   Press R to play again, Esc to quit.")
        elif turn == HUMAN:
            draw_status(screen, font, "Your move — click a cell.")
        else:
            draw_status(screen, font, "TensorLang is thinking...")

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()

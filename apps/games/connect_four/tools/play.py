#!/usr/bin/env python3
"""
Interactive Connect Four: you vs. the TensorLang-trained policy+value
network — or sit back and watch it play itself.

Run from the tensor-lang repo root (after run.sh has trained the
network at least once, or even before — see the fallback note below):

    python3 apps/games/connect_four/tools/play.py

Controls (start screen):
    1     play as Player 1 (red, moves first)
    2     play as Player 2 (blue, moves second)
    S     watch the network play itself (samples from its own policy,
          since always taking its single best move would replay the
          same game every time)
    ESC/Q quit

During a game:
    Click a column   drop a disc there, on your turn (human modes only)
    R                back to the start screen
    ESC/Q            quit

Every AI move runs apps/games/connect_four/infer.tl as a TensorLang
subprocess via tools/agent.py. If that fails for ANY reason — most
importantly, if you haven't trained/promoted a network yet — agent.py
transparently falls back to the plain-Python alpha-beta HeuristicSolver
instead of crashing, so the game is fully playable (and a genuinely
tough opponent) from the very first run. The status bar always shows
which one actually answered the last move.
"""
import sys
import time
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import stats  # noqa: E402
from engine_bitboard import Game, WIDTH, HEIGHT  # noqa: E402

CELL = 90
MARGIN = 20
STATUS_H = 120
WINDOW_W = MARGIN * 2 + WIDTH * CELL
WINDOW_H = MARGIN * 2 + HEIGHT * CELL + STATUS_H

BG = (20, 60, 140)
HOLE_EMPTY = (240, 240, 235)
P1_COLOR = (210, 55, 55)     # red
P2_COLOR = (40, 110, 220)    # blue
STATUS_BG = (245, 241, 230)
TEXT_COLOR = (40, 38, 34)
HINT_COLOR = (110, 106, 100)
WIN_RING = (30, 200, 90)

HUMAN, AI = "human", "ai"

SELF_PLAY_MOVE_DELAY_MS = 400
SELF_PLAY_RESULT_PAUSE_MS = 2000


def board_pos(row, col):
    """Pixel center of a cell. row 0 = top (matches Game.grid())."""
    x = MARGIN + col * CELL + CELL // 2
    y = MARGIN + row * CELL + CELL // 2
    return x, y


def col_at_pos(pos):
    x, y = pos
    if not (MARGIN <= x < MARGIN + WIDTH * CELL and MARGIN <= y < MARGIN + HEIGHT * CELL):
        return None
    return int((x - MARGIN) // CELL)


def winning_cells(grid):
    """Returns a list of (row, col) for a completed 4-in-a-row, or [] if
    the grid has no winner (brute-force scan — board is tiny, 6x7)."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(HEIGHT):
        for c in range(WIDTH):
            val = grid[r][c]
            if val == 0:
                continue
            for dr, dc in directions:
                cells = [(r + dr * i, c + dc * i) for i in range(4)]
                if all(0 <= rr < HEIGHT and 0 <= cc < WIDTH and grid[rr][cc] == val for rr, cc in cells):
                    return cells
    return []


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Connect Four — TensorLang")
        self.font = pygame.font.SysFont("arial", 22)
        self.font_small = pygame.font.SysFont("arial", 16)
        self.font_big = pygame.font.SysFont("arial", 34, bold=True)
        self.clock = pygame.time.Clock()
        self.state = "start"
        self.stats = stats.load_stats()
        self.reset_game()

    def reset_game(self):
        self.game = Game()
        self.players = None  # {0: HUMAN/AI, 1: HUMAN/AI}
        self.last_info = None
        self.last_move_time_ms = 0
        self.self_play_timer = 0
        self.result_timer = 0
        self.result_recorded = False

    def stats_mode(self):
        return "ai_self_play" if self.players == {0: AI, 1: AI} else "human_vs_ai"

    def maybe_record_result(self):
        """Records the just-finished game's outcome exactly once (guarded
        by result_recorded) into apps/games/connect_four/data/play_stats.json.
        """
        if self.result_recorded:
            return
        if self.game.winner is None and not self.game.is_draw():
            return
        self.result_recorded = True
        self.stats = stats.record_result(self.game.winner, self.stats_mode())

    # ---- start screen ----

    def draw_start(self):
        self.screen.fill(STATUS_BG)
        title = self.font_big.render("Connect Four", True, TEXT_COLOR)
        self.screen.blit(title, title.get_rect(center=(WINDOW_W // 2, 60)))
        lines = [
            "1 - Play as Player 1 (red, moves first)",
            "2 - Play as Player 2 (blue, moves second)",
            "S - Watch the network play itself",
            "ESC / Q - Quit",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, surf.get_rect(center=(WINDOW_W // 2, 130 + i * 34)))

        s = self.stats
        tally = (f"All-time: Red {s['red_wins']} - Blue {s['blue_wins']} - "
                 f"Draws {s['draws']}  ({s['games_played']} games)")
        surf = self.font_small.render(tally, True, HINT_COLOR)
        self.screen.blit(surf, surf.get_rect(center=(WINDOW_W // 2, 130 + len(lines) * 34 + 20)))

    def handle_start_key(self, key):
        if key == pygame.K_1:
            self.players = {0: HUMAN, 1: AI}
            self.state = "play"
        elif key == pygame.K_2:
            self.players = {0: AI, 1: HUMAN}
            self.state = "play"
        elif key == pygame.K_s:
            self.players = {0: AI, 1: AI}
            self.state = "play"

    # ---- game screen ----

    def draw_board(self):
        self.maybe_record_result()
        self.screen.fill(STATUS_BG)
        pygame.draw.rect(self.screen, BG, (0, 0, WINDOW_W, HEIGHT * CELL + 2 * MARGIN))
        grid = self.game.grid()
        win_cells = winning_cells(grid) if self.game.winner is not None else []
        for r in range(HEIGHT):
            for c in range(WIDTH):
                x, y = board_pos(r, c)
                val = grid[r][c]
                color = HOLE_EMPTY if val == 0 else (P1_COLOR if val == 1 else P2_COLOR)
                pygame.draw.circle(self.screen, color, (x, y), CELL // 2 - 8)
                if (r, c) in win_cells:
                    pygame.draw.circle(self.screen, WIN_RING, (x, y), CELL // 2 - 8, 4)
        self.draw_status()

    def draw_status(self):
        y0 = HEIGHT * CELL + 2 * MARGIN
        pygame.draw.rect(self.screen, STATUS_BG, (0, y0, WINDOW_W, STATUS_H))
        if self.game.winner is not None:
            label = "Player 1 (red)" if self.game.winner == 0 else "Player 2 (blue)"
            msg = f"{label} wins!  (R = restart)"
        elif self.game.is_draw():
            msg = "Draw!  (R = restart)"
        else:
            turn_label = "Player 1 (red)" if self.game.current_player == 0 else "Player 2 (blue)"
            kind = self.players[self.game.current_player]
            msg = f"{turn_label} to move" + ("" if kind == HUMAN else " (thinking...)")
        surf = self.font.render(msg, True, TEXT_COLOR)
        self.screen.blit(surf, (MARGIN, y0 + 10))

        if self.last_info is not None:
            src = self.last_info.get("source")
            if src == "network":
                value = self.last_info.get("value", 0.0)
                detail = f"last move: network (value {value:+.2f}), {self.last_move_time_ms}ms"
            elif src == "tactical_override":
                reason = self.last_info.get("reason", "")
                label = "immediate win" if reason == "immediate_win" else "forced block"
                detail = f"last move: tactical override ({label})"
            else:
                score = self.last_info.get("score")
                detail = f"last move: solver fallback (score {score}), {self.last_move_time_ms}ms"
            surf2 = self.font_small.render(detail, True, HINT_COLOR)
            self.screen.blit(surf2, (MARGIN, y0 + 42))

        hint = self.font_small.render("R restart   ESC/Q quit", True, HINT_COLOR)
        self.screen.blit(hint, (MARGIN, y0 + 66))

        s = self.stats
        tally = (f"All-time: Red {s['red_wins']} - Blue {s['blue_wins']} - "
                 f"Draws {s['draws']}  ({s['games_played']} games)")
        surf3 = self.font_small.render(tally, True, HINT_COLOR)
        self.screen.blit(surf3, (MARGIN, y0 + 90))

    def ai_move(self):
        mode = "sample" if self.players[0] == AI and self.players[1] == AI else "best"
        t0 = time.monotonic()
        col, info = agent.choose_move(self.game, mode=mode)
        self.last_move_time_ms = int((time.monotonic() - t0) * 1000)
        self.last_info = info
        self.game.play(col)

    def handle_click(self, pos):
        if self.game.winner is not None or self.game.is_draw():
            return
        if self.players[self.game.current_player] != HUMAN:
            return
        col = col_at_pos(pos)
        if col is not None and self.game.can_play(col):
            self.game.play(col)
            self.last_info = None

    def update_self_play(self, dt_ms):
        if self.game.winner is not None or self.game.is_draw():
            self.result_timer += dt_ms
            if self.result_timer > SELF_PLAY_RESULT_PAUSE_MS:
                self.reset_game()
                self.players = {0: AI, 1: AI}
            return
        self.self_play_timer += dt_ms
        if self.self_play_timer >= SELF_PLAY_MOVE_DELAY_MS:
            self.self_play_timer = 0
            self.ai_move()

    def run(self):
        running = True
        while running:
            dt_ms = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif self.state == "start":
                        self.handle_start_key(event.key)
                    elif self.state == "play" and event.key == pygame.K_r:
                        self.reset_game()
                        self.state = "start"
                elif event.type == pygame.MOUSEBUTTONDOWN and self.state == "play":
                    self.handle_click(event.pos)

            if self.state == "start":
                self.draw_start()
            else:
                if self.players[0] == AI and self.players[1] == AI:
                    self.update_self_play(dt_ms)
                elif (self.game.winner is None and not self.game.is_draw()
                        and self.players[self.game.current_player] == AI):
                    self.ai_move()
                self.draw_board()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    App().run()

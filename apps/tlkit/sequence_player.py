"""
Reusable pygame play/pause/scrub/speed harness for frame-sequence
animations — the kind tlkit.chunked_runner.run_chunks() produces, or
any (F, ...) numpy array of frames from a one-shot save() (e.g. GPU
physics playback, no chunking needed).

Usage:
    from tlkit.sequence_player import SequencePlayer

    def render(frame):            # frame -> pygame.Surface
        ...
    def overlay(screen, index):   # optional: extra per-frame drawing
        ...

    SequencePlayer(frames, render, overlay_fn=overlay,
                   caption="...").run()

Controls: SPACE play/pause, LEFT/RIGHT step, UP/DOWN speed, ESC/Q quit.
"""
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    import pygame


class SequencePlayer:
    def __init__(
        self,
        frames: np.ndarray,
        render_fn: Callable[[np.ndarray], "pygame.Surface"],
        overlay_fn: Optional[Callable[["pygame.Surface", int], None]] = None,
        window_size: int = 600,
        caption: str = "TensorLang playback",
        fps: int = 12,
    ):
        if len(frames) == 0:
            raise ValueError("frames is empty — nothing to play")
        self.frames = frames
        self.render_fn = render_fn
        self.overlay_fn = overlay_fn
        self.window_size = window_size
        self.caption = caption
        self.fps = fps
        self.index = 0
        self.playing = True

    def run(self) -> None:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption(self.caption)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key, running)

            surf = self.render_fn(self.frames[self.index])
            screen.blit(surf, (0, 0))
            if self.overlay_fn:
                self.overlay_fn(screen, self.index)

            label = font.render(f"frame {self.index + 1}/{len(self.frames)}", True, (20, 20, 20))
            screen.blit(label, (10, 10))
            pygame.display.flip()

            if self.playing:
                self.index = (self.index + 1) % len(self.frames)
            clock.tick(self.fps)

        pygame.quit()

    def _handle_key(self, key: int, running: bool) -> bool:
        import pygame

        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        elif key == pygame.K_SPACE:
            self.playing = not self.playing
        elif key == pygame.K_RIGHT:
            self.index = min(self.index + 1, len(self.frames) - 1)
        elif key == pygame.K_LEFT:
            self.index = max(self.index - 1, 0)
        elif key == pygame.K_UP:
            self.fps = min(self.fps + 2, 60)
        elif key == pygame.K_DOWN:
            self.fps = max(self.fps - 2, 1)
        return running

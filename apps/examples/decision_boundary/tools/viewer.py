#!/usr/bin/env python3
"""
pygame viewer for the decision-boundary animation produced by
tools/train_and_snapshot.py.

Run from anywhere (this locates the repo root relative to itself):

    python3 apps/examples/decision_boundary/tools/viewer.py

Controls:
    SPACE        play / pause
    LEFT / RIGHT step one frame back / forward
    UP / DOWN    speed up / slow down playback
    ESC or Q     quit
"""
import json
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[4]
APP_DIR = ROOT / "apps/examples/decision_boundary"
FRAMES_PATH = APP_DIR / "snapshots/frames.npy"
META_PATH = APP_DIR / "data/meta.json"

WINDOW_SIZE = 600
DOT_RADIUS = 9

# Diverging colormap: blue (class 0) -> white (uncertain, p=0.5) -> red (class 1)
LOW = np.array([40, 70, 220], dtype=np.float32)
MID = np.array([245, 245, 245], dtype=np.float32)
HIGH = np.array([220, 55, 55], dtype=np.float32)


def colormap(p: np.ndarray) -> np.ndarray:
    """p: (H, W) float32 in [0,1] -> (H, W, 3) uint8"""
    p = np.clip(p, 0.0, 1.0)
    t_low = np.clip(p * 2.0, 0.0, 1.0)[..., None]
    lower = LOW * (1 - t_low) + MID * t_low
    t_high = np.clip((p - 0.5) * 2.0, 0.0, 1.0)[..., None]
    upper = MID * (1 - t_high) + HIGH * t_high
    color = np.where((p < 0.5)[..., None], lower, upper)
    return color.astype(np.uint8)


def build_frame_surface(frame: np.ndarray) -> pygame.Surface:
    """frame: (side, side) float32 probabilities, row=y index, col=x index."""
    color = colormap(frame)          # (side, side, 3), [y, x]
    color = np.flipud(color)         # so larger y renders higher on screen
    color = color.transpose(1, 0, 2) # -> [x, y, 3], what surfarray expects
    surf = pygame.surfarray.make_surface(color)
    return pygame.transform.smoothscale(surf, (WINDOW_SIZE, WINDOW_SIZE))


def data_to_pixel(x, y, lo, hi):
    px = (x - lo) / (hi - lo) * WINDOW_SIZE
    py = WINDOW_SIZE - (y - lo) / (hi - lo) * WINDOW_SIZE  # flip to match build_frame_surface
    return int(px), int(py)


def main():
    if not FRAMES_PATH.exists():
        raise SystemExit(
            f"No {FRAMES_PATH} found.\n"
            "Run init_state.py then train_and_snapshot.py first."
        )
    frames = np.load(FRAMES_PATH)  # (num_frames, side, side)
    meta = json.loads(META_PATH.read_text())
    lo, hi = meta["extent"]
    xor_points = meta["xor_points"]
    xor_labels = meta["xor_labels"]

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("TensorLang: XOR decision boundary")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    i = 0
    playing = True
    fps = 12

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    i = min(i + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    i = max(i - 1, 0)
                elif event.key == pygame.K_UP:
                    fps = min(fps + 2, 60)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps - 2, 1)

        surf = build_frame_surface(frames[i])
        screen.blit(surf, (0, 0))

        for (x, y), label in zip(xor_points, xor_labels):
            px, py = data_to_pixel(x, y, lo, hi)
            color = (40, 70, 220) if label == 0 else (220, 55, 55)
            pygame.draw.circle(screen, color, (px, py), DOT_RADIUS)
            pygame.draw.circle(screen, (20, 20, 20), (px, py), DOT_RADIUS, width=2)

        label_surf = font.render(f"chunk {i+1}/{len(frames)}", True, (20, 20, 20))
        screen.blit(label_surf, (10, 10))

        pygame.display.flip()

        if playing:
            i = (i + 1) % len(frames)
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()

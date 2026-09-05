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
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pygame

# from apps/examples/decision_boundary/tools/viewer.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner, colormap, sequence_player  # noqa: E402

APP_DIR = Path("apps/examples/decision_boundary")
WINDOW_SIZE = 600
DOT_RADIUS = 9


def draw_xor_overlay(meta, screen, _frame_index):
    lo, hi = meta["extent"]
    for (x, y), label in zip(meta["xor_points"], meta["xor_labels"]):
        px, py = colormap.data_to_pixel(x, y, lo, hi, WINDOW_SIZE)
        color = (40, 70, 220) if label == 0 else (220, 55, 55)
        pygame.draw.circle(screen, color, (px, py), DOT_RADIUS)
        pygame.draw.circle(screen, (20, 20, 20), (px, py), DOT_RADIUS, width=2)


def main():
    repo_root = chunked_runner.find_repo_root()
    frames_path = repo_root / APP_DIR / "snapshots/frames.npy"
    meta_path = repo_root / APP_DIR / "data/meta.json"

    if not frames_path.exists():
        raise SystemExit(
            f"No {frames_path} found.\nRun init_state.py then train_and_snapshot.py first."
        )
    frames = np.load(frames_path)  # (num_frames, side, side)
    meta = json.loads(meta_path.read_text())

    player = sequence_player.SequencePlayer(
        frames,
        render_fn=lambda f: colormap.grid_to_surface(f, WINDOW_SIZE),
        overlay_fn=partial(draw_xor_overlay, meta),
        window_size=WINDOW_SIZE,
        caption="TensorLang: XOR decision boundary",
    )
    player.run()


if __name__ == "__main__":
    main()
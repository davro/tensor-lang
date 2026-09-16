#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training run:

    python3 apps/games/snake/tools/init_weights.py

Creates a fresh random init for the 720-256-64-4 move-picking policy
network at:
    cache/apps/games/snake/train.tl/weights/{w1,b1,w2,b2,w3,b3}.npy

720 inputs = 12x12 board cells x 5 one-hot categories (see
tools/agent.py's encode_state_onehot). 4 outputs = one logit per
direction, in agent.DIRECTIONS order (up, down, left, right).

He/Kaiming-style init (std = sqrt(2/fan_in)), same reasoning as 2048's
init_weights.py: two hidden layers, and a wide one-hot input where only
GRID_W*GRID_H of 720 features are ever nonzero per row, so scaling each
layer's init by its own fan-in keeps pre-activations sane regardless of
that width.

Re-run this any time to wipe training progress and start fresh (it
always overwrites, same as every other app's init_weights.py).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

INPUT = agent.INPUT_DIM
HIDDEN1 = 256
HIDDEN2 = 64
OUTPUT = 4
SEED = 0


def he_init(rng, fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return (rng.standard_normal((fan_in, fan_out)) * std).astype(np.float32)


def main():
    repo_root = chunked_runner.find_repo_root()
    weights_dir = repo_root / "cache" / "apps" / "games" / "snake" / "train.tl" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    w1 = he_init(rng, INPUT, HIDDEN1)
    b1 = np.zeros((HIDDEN1,), dtype=np.float32)
    w2 = he_init(rng, HIDDEN1, HIDDEN2)
    b2 = np.zeros((HIDDEN2,), dtype=np.float32)
    w3 = he_init(rng, HIDDEN2, OUTPUT)
    b3 = np.zeros((OUTPUT,), dtype=np.float32)

    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2), ("w3", w3), ("b3", b3)]:
        np.save(weights_dir / f"{name}.npy", arr)

    print(f"Wrote fresh random weights to {weights_dir}")
    print(f"  w1 {w1.shape}  b1 {b1.shape}  w2 {w2.shape}  b2 {b2.shape}  w3 {w3.shape}  b3 {b3.shape}")


if __name__ == "__main__":
    main()

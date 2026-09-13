#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training run:

    python3 apps/games/connect_four/tools/init_weights.py

Creates a fresh random init for the 42-128-64 shared trunk plus a
7-column policy head and a 1-unit value head, at:
  cache/apps/games/connect_four/train.tl/weights/{w1,b1,w2,b2,wp,bp,wv,bv}.npy

Re-run this (or use run.sh --reset) to wipe training progress and start
from scratch. This does NOT touch apps/games/connect_four/weights/ (the
promoted, tracked, "production" copy infer.tl reads from) — see
promote_weights.py for that step.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # apps/
from tlkit import chunked_runner  # noqa: E402

IN_DIM = 42
H1 = 128
H2 = 64
POLICY_OUT = 7
VALUE_OUT = 2  # 2 units, not 1 -- see train.tl's "REAL BUG #3" note
SEED = 0


def _he_init(rng, fan_in, shape):
    """He/Kaiming-style init, appropriate for ReLU trunk layers."""
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def main():
    repo_root = chunked_runner.find_repo_root()
    weights_dir = repo_root / "cache" / "apps" / "games" / "connect_four" / "train.tl" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    w1 = _he_init(rng, IN_DIM, (IN_DIM, H1))
    b1 = np.zeros((H1,), dtype=np.float32)
    w2 = _he_init(rng, H1, (H1, H2))
    b2 = np.zeros((H2,), dtype=np.float32)
    # Smaller init on the heads (fan-in H2, but scaled down further since
    # softmax/tanh outputs are sensitive to large pre-activations early on)
    wp = (rng.standard_normal((H2, POLICY_OUT)) * 0.1).astype(np.float32)
    bp = np.zeros((POLICY_OUT,), dtype=np.float32)
    wv = (rng.standard_normal((H2, VALUE_OUT)) * 0.1).astype(np.float32)
    bv = np.zeros((VALUE_OUT,), dtype=np.float32)

    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                       ("wp", wp), ("bp", bp), ("wv", wv), ("bv", bv)]:
        np.save(weights_dir / f"{name}.npy", arr)

    print(f"Wrote fresh random weights to {weights_dir}")
    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                       ("wp", wp), ("bp", bp), ("wv", wv), ("bv", bv)]:
        print(f"  {name} {arr.shape}")


if __name__ == "__main__":
    main()

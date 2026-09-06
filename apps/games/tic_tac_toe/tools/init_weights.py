#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training run:

    python3 apps/games/tic_tac_toe/tools/init_weights.py

Creates a fresh random init for the 9-32-9 policy network at:
  cache/apps/games/tic_tac_toe/train.tl/weights/{w1,b1,w2,b2}.npy

Re-run this (or use run.sh --reset) to wipe training progress and start
from scratch.
"""
import sys
from pathlib import Path

import numpy as np

# from apps/games/tic_tac_toe/tools/init_weights.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

HIDDEN = 64
SEED = 0


def main():
    repo_root = chunked_runner.find_repo_root()
    # chunked_runner.cache_dir_for() assumes the `--app`-runner convention
    # of a fixed "main.tl" entry point; this app trains via a bare
    # `train.tl`, invoked directly rather than through `--app`, so the
    # cache path is built explicitly here instead (it mirrors
    # compiler.py's own cache_base / file_path logic: cache/<file path>/).
    weights_dir = repo_root / "cache" / "apps" / "games" / "tic_tac_toe" / "train.tl" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    w1 = (rng.standard_normal((9, HIDDEN)) * 0.4).astype(np.float32)
    b1 = np.zeros((HIDDEN,), dtype=np.float32)
    w2 = (rng.standard_normal((HIDDEN, 9)) * 0.4).astype(np.float32)
    b2 = np.zeros((9,), dtype=np.float32)

    np.save(weights_dir / "w1.npy", w1)
    np.save(weights_dir / "b1.npy", b1)
    np.save(weights_dir / "w2.npy", w2)
    np.save(weights_dir / "b2.npy", b2)

    print(f"Wrote fresh random weights to {weights_dir}")
    print(f"  w1 {w1.shape}  b1 {b1.shape}  w2 {w2.shape}  b2 {b2.shape}")


if __name__ == "__main__":
    main()

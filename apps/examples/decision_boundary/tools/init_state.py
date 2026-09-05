#!/usr/bin/env python3
"""
Run ONCE, from the tensor-lang repo root, before the first training chunk:

    python3 apps/examples/decision_boundary/tools/init_state.py

Creates:
  - cache/apps/examples/decision_boundary/main.tl/weights/w1.npy  (2, 8)
  - cache/apps/examples/decision_boundary/main.tl/weights/w2.npy  (8, 1)
  - apps/examples/decision_boundary/data/grid.npy                (1600, 2)
  - apps/examples/decision_boundary/data/meta.json                (viewer metadata)

Re-run this to reset training back to a fresh random init.
"""
import json
import sys
from pathlib import Path

import numpy as np

# from apps/examples/decision_boundary/tools/init_state.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner, colormap  # noqa: E402

REPO_ROOT = chunked_runner.find_repo_root()
APP_DIR = Path("apps/examples/decision_boundary")

SIDE = 40             # grid resolution: SIDE x SIDE points
EXTENT = (-0.5, 1.5)  # both x and y range over this, so the XOR corners sit inside the frame
HIDDEN = 8
SEED = 0


def main():
    rng = np.random.default_rng(SEED)

    weights_dir = chunked_runner.cache_dir_for("examples/decision_boundary", REPO_ROOT) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    w1 = (rng.standard_normal((2, HIDDEN)) * 0.5).astype(np.float32)
    w2 = (rng.standard_normal((HIDDEN, 1)) * 0.5).astype(np.float32)
    b1 = np.zeros((HIDDEN,), dtype=np.float32)
    b2 = np.zeros((1,), dtype=np.float32)
    np.save(weights_dir / "w1.npy", w1)
    np.save(weights_dir / "w2.npy", w2)
    np.save(weights_dir / "b1.npy", b1)
    np.save(weights_dir / "b2.npy", b2)
    print(f"Wrote initial weights: {weights_dir/'w1.npy'} {w1.shape}, {weights_dir/'w2.npy'} {w2.shape}")
    print(f"Wrote initial biases:  {weights_dir/'b1.npy'} {b1.shape}, {weights_dir/'b2.npy'} {b2.shape}")

    data_dir = REPO_ROOT / APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    grid = colormap.make_grid(SIDE, EXTENT)  # (SIDE*SIDE, 2), matches main.tl's Tensor[f32,(1600,2)]
    np.save(data_dir / "grid.npy", grid)
    print(f"Wrote evaluation grid: {data_dir/'grid.npy'} {grid.shape}")

    meta = {
        "side": SIDE,
        "extent": list(EXTENT),
        "xor_points": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        "xor_labels": [0, 1, 1, 0],
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata: {data_dir/'meta.json'}")


if __name__ == "__main__":
    main()
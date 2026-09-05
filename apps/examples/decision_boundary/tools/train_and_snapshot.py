#!/usr/bin/env python3
"""
Run from the tensor-lang repo root (after init_state.py, and after
`source python-env/bin/activate`):

    python3 apps/examples/decision_boundary/tools/train_and_snapshot.py

Runs main.tl 40 times (40 chunks x 25 epochs/chunk = 1000 total epochs
— see the `for epoch in range(25)` loop in main.tl), resuming from the
weights the previous chunk saved each time. After each chunk, reads
back the grid snapshot main.tl just wrote and stacks all of them into
apps/examples/decision_boundary/snapshots/frames.npy, shape
(num_chunks, side, side) — exactly what viewer.py expects.
"""
import json
import sys
from pathlib import Path

import numpy as np

# from apps/examples/decision_boundary/tools/train_and_snapshot.py, parents[3] is apps/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

APP = "examples/decision_boundary"
APP_DIR = Path("apps/examples/decision_boundary")
NUM_CHUNKS = 40


def collect_frame(repo_root: Path) -> np.ndarray:
    side = json.loads((repo_root / APP_DIR / "data/meta.json").read_text())["side"]
    frame_path = chunked_runner.cache_dir_for(APP, repo_root) / "frames" / "frame.npy"
    return np.load(frame_path).reshape(side, side)  # (side*side, 1) -> (side, side)


def report_progress(chunk: int, n_chunks: int) -> str:
    repo_root = chunked_runner.find_repo_root()
    loss_path = chunked_runner.cache_dir_for(APP, repo_root) / "frames" / "loss.npy"
    loss = np.load(loss_path).item()  # mse_loss saves a (1,1) tensor, not a bare scalar
    return f"chunk {chunk + 1:3d}/{n_chunks}   loss_final={loss:.6f}"


def main():
    repo_root = chunked_runner.find_repo_root()
    meta_path = repo_root / APP_DIR / "data/meta.json"
    if not meta_path.exists():
        raise SystemExit("Run tools/init_state.py first (no data/meta.json found).")

    frames = chunked_runner.run_chunks(APP, NUM_CHUNKS, collect_frame, repo_root, report_progress)

    out_path = chunked_runner.save_frames(frames, repo_root / APP_DIR / "snapshots/frames.npy")
    print(f"\nSaved {frames.shape} to {out_path}")
    print("Now run: python3 apps/examples/decision_boundary/tools/viewer.py")


if __name__ == "__main__":
    main()
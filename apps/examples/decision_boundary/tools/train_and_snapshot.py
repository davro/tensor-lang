#!/usr/bin/env python3
"""
Run from the tensor-lang repo root (after init_state.py, and after
`source python-env/bin/activate`):

    python3 apps/examples/decision_boundary/tools/train_and_snapshot.py

Repeatedly invokes:

    python3 tensorlang.py --app examples/decision_boundary

Each invocation runs one 25-epoch training chunk (main.tl), resuming from
the weights the previous chunk saved. After each chunk this script reads
back the grid snapshot main.tl just wrote and appends it to an in-memory
list. At the end it stacks everything into one array and saves it as
apps/examples/decision_boundary/snapshots/frames.npy, shape
(num_chunks, side, side) — exactly what viewer.py expects.
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[4]
APP_DIR     = Path("apps/examples/decision_boundary")
CACHE_DIR   = Path("cache/apps/examples/decision_boundary/main.tl")
FRAME_FILE  = REPO_ROOT / CACHE_DIR / "frames" / "frame.npy"
LOSS_FILE   = REPO_ROOT / CACHE_DIR / "frames" / "loss.npy"

NUM_CHUNKS = 40  # 40 chunks x 25 epochs/chunk = 1000 total epochs


def run_one_chunk():
    result = subprocess.run(
        [sys.executable, "tensorlang.py", "--app", "examples/decision_boundary"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("tensorlang.py invocation failed — see output above")


def main():
    meta_path = REPO_ROOT / APP_DIR / "data" / "meta.json"
    if not meta_path.exists():
        raise SystemExit("Run tools/init_state.py first (no data/meta.json found).")
    meta = json.loads(meta_path.read_text())
    side = meta["side"]

    frames = []
    for chunk in range(NUM_CHUNKS):
        run_one_chunk()
        frame = np.load(FRAME_FILE).reshape(side, side)  # (side*side, 1) -> (side, side)
        loss = float(np.load(LOSS_FILE))
        frames.append(frame)
        print(f"chunk {chunk+1:3d}/{NUM_CHUNKS}   loss_final={loss:.6f}")

    stacked = np.stack(frames, axis=0).astype(np.float32)  # (NUM_CHUNKS, side, side)

    out_dir = REPO_ROOT / APP_DIR / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frames.npy"
    np.save(out_path, stacked)
    print(f"\nSaved {stacked.shape} to {out_path}")
    print("Now run: python3 apps/examples/decision_boundary/tools/viewer.py")


if __name__ == "__main__":
    main()

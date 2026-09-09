#!/usr/bin/env python3
"""Write a test board into the cache location step.tl expects, so you can
run:  python3 tensorlang.py apps/games/2048/step.tl
and then inspect cache/apps/games/2048/step.tl/new_board.npy afterwards.

Usage:
  python3 make_test_board.py                      # uses the built-in tricky test board
  python3 make_test_board.py 4 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0   # 16 numbers, row-major
"""
import sys
import numpy as np
from pathlib import Path

if len(sys.argv) == 17:
    vals = [float(x) for x in sys.argv[1:]]
    board = np.array(vals, dtype=np.float32).reshape(4, 4)
else:
    # The [4,2,2,0] counter-example from step.tl's comments, in row 0,
    # plus a couple of other rows so you can see multiple rows moving.
    board = np.array([
        [4, 2, 2, 0],
        [2, 2, 2, 2],
        [0, 2, 0, 2],
        [8, 4, 2, 2],
    ], dtype=np.float32)

out_dir = Path("cache/apps/games/2048/step.tl")
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "board.npy", board)
print(f"Wrote {out_dir / 'board.npy'}:")
print(board)
print("\nNow run:  python3 tensorlang.py apps/games/2048/step.tl")
print(f"Then check:  {out_dir / 'new_board.npy'}")

#!/usr/bin/env python3
"""Write a test board into the cache location EACH step_*.tl file expects,
so you can run any one of them directly, e.g.:
  python3 tensorlang.py apps/games/2048/step.tl
  python3 tensorlang.py apps/games/2048/step_right.tl
  python3 tensorlang.py apps/games/2048/step_up.tl
  python3 tensorlang.py apps/games/2048/step_down.tl
and then inspect cache/apps/games/2048/<file>/new_board.npy afterwards.

Each of the four step_*.tl files load()s from its OWN cache subdirectory
(cache/apps/games/2048/step.tl/, .../step_right.tl/, etc. — see each
file's `load(...)` line) rather than a shared one, since that's just the
literal path string baked into each program (tensor-lang's load() paths
aren't auto-derived from the script name — see app_runner.py). This script
writes the SAME board into all four, so you can compare one board's
left/right/up/down results side by side without re-running this each time
you switch which direction you're testing.

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

STEP_FILES = ["step.tl", "step_right.tl", "step_up.tl", "step_down.tl"]

for step_file in STEP_FILES:
    out_dir = Path("cache/apps/games/2048") / step_file
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "board.npy", board)
    print(f"Wrote {out_dir / 'board.npy'}")

print()
print(board)
print("\nNow run any of:")
for step_file in STEP_FILES:
    print(f"  python3 tensorlang.py apps/games/2048/{step_file}")
print("Then check the matching cache/apps/games/2048/<file>/new_board.npy")

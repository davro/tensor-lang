#!/usr/bin/env python3
"""
Undo the last promote_weights.py: swaps apps/games/2048/weights/ with
apps/games/2048/weights/.previous/.

    python3 apps/games/2048/tools/rollback_weights.py
    # or: ./apps/games/2048/run.sh --rollback

This is a SWAP, not a one-directional restore — running it twice in a
row gets you back to where you started, rather than losing the "current"
version you rolled back from. It's a single slot either way (see
promote_weights.py's docstring): this covers "that last training run was
worse, go back one step," not a full history. Once a promoted set of
weights is actually committed to git, git itself is the real history —
this script only helps for the pre-commit window.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

WEIGHT_FILES = ["w1.npy", "b1.npy", "w2.npy", "b2.npy", "w3.npy", "b3.npy"]


def main():
    repo_root = chunked_runner.find_repo_root()
    current_dir = repo_root / "apps" / "games" / "2048" / "weights"
    backup_dir = current_dir / ".previous"

    if not backup_dir.exists() or not all((backup_dir / f).exists() for f in WEIGHT_FILES):
        print(f"ERROR: no complete backup found at {backup_dir} — nothing to roll back to.")
        sys.exit(1)

    tmp_dir = current_dir / ".rollback_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for f in WEIGHT_FILES:
        (current_dir / f).rename(tmp_dir / f)          # current -> tmp
    for f in WEIGHT_FILES:
        (backup_dir / f).rename(current_dir / f)        # previous -> current
    for f in WEIGHT_FILES:
        (tmp_dir / f).rename(backup_dir / f)             # tmp -> previous
    tmp_dir.rmdir()

    print(f"Swapped {current_dir} <-> {backup_dir}.")
    print("Run this again to swap back if you change your mind.")
    print("infer.tl will use the restored weights from now on.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Promote the weights train.tl just produced into the tracked, in-repo
"production" location that infer.tl actually loads from:

    cache/apps/games/snake/train.tl/weights/*.npy   (train.tl's scratch output)
        --promote-->
    apps/games/snake/weights/*.npy                  (tracked; infer.tl reads this)

Run after a training run you're happy with:

    python3 apps/games/snake/tools/promote_weights.py
    # or: ./apps/games/snake/run.sh --promote

See apps/games/2048/tools/promote_weights.py for the full reasoning
(identical here): cache/ is scratch and untracked, a trained network
represents real GPU time, and this is the one deliberate checkpoint
action that separates "training experiment" from "what the game
actually uses now."

BACKUP: if apps/games/snake/weights/ already has a promoted set, it's
moved to apps/games/snake/weights/.previous/ first (a single rollback
slot, overwriting any earlier backup) — see rollback_weights.py.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tlkit import chunked_runner  # noqa: E402

WEIGHT_FILES = ["w1.npy", "b1.npy", "w2.npy", "b2.npy", "w3.npy", "b3.npy"]


def main():
    repo_root = chunked_runner.find_repo_root()
    src_dir = repo_root / "cache" / "apps" / "games" / "snake" / "train.tl" / "weights"
    dst_dir = repo_root / "apps" / "games" / "snake" / "weights"
    backup_dir = dst_dir / ".previous"

    missing = [f for f in WEIGHT_FILES if not (src_dir / f).exists()]
    if missing:
        print(f"ERROR: {src_dir} is missing {missing} — run ./run.sh --train first.")
        sys.exit(1)

    existing = [f for f in WEIGHT_FILES if (dst_dir / f).exists()]
    if existing:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in existing:
            shutil.copy2(dst_dir / f, backup_dir / f)
        print(f"Backed up current {dst_dir} -> {backup_dir} "
              f"(overwrote any earlier backup — single rollback slot).")
    else:
        print(f"{dst_dir} is empty (first promotion) — nothing to back up.")

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in WEIGHT_FILES:
        shutil.copy2(src_dir / f, dst_dir / f)

    print(f"Promoted {src_dir} -> {dst_dir}")
    print("infer.tl (and therefore agent.py's choose_ai_move) will use these from now on.")
    print(f"If you want this checked in: git add {dst_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

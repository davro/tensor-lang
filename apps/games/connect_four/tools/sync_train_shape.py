#!/usr/bin/env python3
"""
apps/games/connect_four/tools/sync_train_shape.py

train.tl declares its dataset size as a static shape (TensorLang has no
dynamic/runtime shapes) — three lines:
    let x:        Tensor[f32, (N, 42)] = load(".../boards.npy")
    let y_policy: Tensor[f32, (N, 7)]  = load(".../policy.npy")
    let y_value:  Tensor[f32, (N, 2)]  = load(".../value.npy")
Whenever generate_data.py produces a dataset of a different size N (e.g.
run.sh --train --num-positions 50000), these three declarations must be
updated to match, or check.py's type checker rejects the shape mismatch
against the actual .npy files on disk.

This reads the ACTUAL N from data/meta.json (written by generate_data.py
itself — ground truth, not re-derived or guessed), and rewrites exactly
those three declarations in train.tl by matching on the variable name
(`let x:`, `let y_policy:`, `let y_value:`), not a blind shape-pattern
regex — several OTHER declarations in the file also contain small
integers matching e.g. `, 7)` (the policy head's width) or `, 2)` (the
value head's width, see NOTES.md's "REAL BUG #3"), so anchoring on the
variable name is what keeps this from corrupting an unrelated line.

Run automatically by run.sh --train whenever --num-positions/--depth/
--time-limit are passed (which force dataset regeneration); safe to run
manually too:
    python3 apps/games/connect_four/tools/sync_train_shape.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # apps/
from tlkit import chunked_runner  # noqa: E402

APP_DIR_REL = Path("apps/games/connect_four")


def main():
    repo_root = chunked_runner.find_repo_root()
    app_dir = repo_root / APP_DIR_REL
    meta_path = app_dir / "data" / "meta.json"
    train_path = app_dir / "train.tl"

    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found — run generate_data.py first.")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    n = meta["num_positions"]

    content = train_path.read_text()
    original = content

    for var_name, width in [("x", 42), ("y_policy", 7), ("y_value", 2)]:
        # Anchored on the variable name so this can't touch wp's (64, 7)
        # or wv's (64, 2) declarations, which share the same trailing
        # width but are never the dataset-size dimension.
        pattern = rf"(let {re.escape(var_name)}:\s*Tensor\[f32, \()\d+(, {width}\)\])"
        new_content, count = re.subn(pattern, rf"\g<1>{n}\g<2>", content)
        if count != 1:
            print(f"ERROR: expected exactly 1 match for `let {var_name}: ... ({width})` "
                  f"in {train_path}, found {count}. Not touching the file — fix manually.")
            sys.exit(1)
        content = new_content

    if content == original:
        print(f"train.tl already declares N={n} — nothing to change.")
        return

    train_path.write_text(content)
    print(f"Updated train.tl's dataset-size declarations (x/y_policy/y_value) to N={n}, "
          f"matching {meta_path}.")


if __name__ == "__main__":
    main()

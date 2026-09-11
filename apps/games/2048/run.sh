#!/usr/bin/env bash
# apps/games/2048/run.sh
#
# Status: the tensor-ops engine (step.tl + step_right.tl/step_up.tl/
# step_down.tl), a pygame UI (tools/agent.py + tools/play.py), and a
# trained move-picking network (train.tl + tools/generate_data.py +
# tools/init_weights.py) all exist now — see apps/games/2048/NOTES.md.
# --play's autoplay uses the trained network via infer.tl, falling back to
# the original hand-written heuristic if the weights aren't there yet or
# inference fails for any reason (run --train first to avoid that).
#
#   ./apps/games/2048/run.sh                 # smoke test: run step.tl on the
#                                             #   built-in tricky test board
#                                             #   (see tools/make_test_board.py)
#   ./apps/games/2048/run.sh --board N N ... # smoke test on a custom board:
#                                             #   16 numbers, row-major, e.g.
#                                             #   --board 4 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0
#   ./apps/games/2048/run.sh --play          # launch the interactive pygame UI
#                                             #   (arrow keys/WASD to play, B to
#                                             #   watch the trained AI play)
#   ./apps/games/2048/run.sh --train         # generate expectimax-labeled
#                                             #   training data (if not already
#                                             #   present) and train the
#                                             #   move-picking network
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"

echo "== checking pygame =="
python3 -c "import pygame" 2>/dev/null || pip install pygame

not_built_yet() {
    local mode_name="$1"
    shift
    local missing=()
    for f in "$@"; do
        [[ -f "$APP_DIR/$f" ]] || missing+=("$f")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "$mode_name isn't built yet — missing:"
        for f in "${missing[@]}"; do
            echo "  - apps/games/2048/$f"
        done
        echo "See apps/games/2048/NOTES.md for what's done and what's next."
        exit 1
    fi
}

if [[ "$MODE" == "--play" ]]; then
    not_built_yet "--play" "tools/agent.py" "tools/play.py"
    echo "== launching game =="
    exec python3 "$APP_DIR/tools/play.py"
fi

if [[ "$MODE" == "--train" ]]; then
    not_built_yet "--train" "train.tl" "tools/generate_data.py" "tools/init_weights.py"
    if [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
        echo "== generating training data (expectimax-labeled board states) =="
        python3 "$APP_DIR/tools/generate_data.py"
    fi
    WEIGHTS_DIR="cache/apps/games/2048/train.tl/weights"
    if [[ ! -d "$WEIGHTS_DIR" ]]; then
        echo "== initializing fresh random weights =="
        python3 "$APP_DIR/tools/init_weights.py"
    fi
    echo "== training =="
    exec python3 tensorlang.py "apps/games/2048/train.tl"
fi

# --- default / --board: smoke-test the engine we actually have (step.tl) ---
not_built_yet "the engine smoke test" "step.tl" "tools/make_test_board.py"

if [[ "$MODE" == "--board" ]]; then
    shift
    echo "== writing custom test board =="
    python3 "$APP_DIR/tools/make_test_board.py" "$@"
else
    echo "== writing built-in test board (see tools/make_test_board.py) =="
    python3 "$APP_DIR/tools/make_test_board.py"
fi

echo "== running step.tl =="
python3 tensorlang.py "apps/games/2048/step.tl"

echo "== result =="
python3 -c "
import numpy as np
print(np.load('cache/apps/games/2048/step.tl/new_board.npy'))
"

#!/usr/bin/env bash
# apps/games/2048/run.sh
#
# Status: only the tensor-ops game engine (step.tl) exists so far — no
# neural net, no pygame UI. So unlike tic_tac_toe's run.sh (which trains a
# policy net then launches an interactive match), this script's default
# action is a smoke test: run step.tl on a real board through the actual
# compiler and show you the before/after. --play and --train are wired in
# ahead of time so the *interface* matches tic_tac_toe's script, but they
# fail with a clear message until their underlying files exist — see
# apps/games/2048/NOTES.md for what's built vs. still to come.
#
#   ./apps/games/2048/run.sh                 # smoke test: run step.tl on the
#                                             #   built-in tricky test board
#                                             #   (see tools/make_test_board.py)
#   ./apps/games/2048/run.sh --board N N ... # smoke test on a custom board:
#                                             #   16 numbers, row-major, e.g.
#                                             #   --board 4 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0
#   ./apps/games/2048/run.sh --play          # launch the interactive game
#                                             #   (not implemented yet — needs
#                                             #   tools/agent.py + tools/play.py)
#   ./apps/games/2048/run.sh --train         # train the move-picking network
#                                             #   (not implemented yet — needs
#                                             #   train.tl + tools/generate_data.py)
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

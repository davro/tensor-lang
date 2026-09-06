#!/usr/bin/env bash
# apps/games/tic_tac_toe/run.sh
#
# Installs pygame if missing, generates the training dataset and a fresh
# weight init on first run, trains the policy network, then launches the
# interactive pygame match. Run from anywhere; paths below are resolved
# relative to this script and the repo root.
#
#   ./apps/games/tic_tac_toe/run.sh          # normal run (trains once, then plays)
#   ./apps/games/tic_tac_toe/run.sh --reset  # wipe weights and retrain from scratch
#   ./apps/games/tic_tac_toe/run.sh --play   # skip training, just launch the game
#                                             #   (fails if no weights exist yet)
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "== checking pygame =="
python3 -c "import pygame" 2>/dev/null || pip install pygame

WEIGHTS_DIR="cache/apps/games/tic_tac_toe/train.tl/weights"

if [[ "${1:-}" == "--play" ]]; then
    if [[ ! -d "$WEIGHTS_DIR" ]]; then
        echo "No trained weights found at $WEIGHTS_DIR — run without --play first."
        exit 1
    fi
    echo "== launching game (skipping training) =="
    exec python3 "$APP_DIR/tools/play.py"
fi

if [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
    echo "== generating training data (minimax-labeled board states) =="
    python3 "$APP_DIR/tools/generate_data.py"
fi

if [[ "${1:-}" == "--reset" || ! -d "$WEIGHTS_DIR" ]]; then
    echo "== initializing fresh random weights =="
    python3 "$APP_DIR/tools/init_weights.py"
else
    echo "== existing weights found, continuing to train from them (use --reset to start over) =="
fi

echo "== training (single run, 12000 epochs over all 4520 reachable board states — takes longer than a quick test run, still under a couple of minutes on a real GPU) =="
python3 tensorlang.py "apps/games/tic_tac_toe/train.tl"

echo "== launching game =="
exec python3 "$APP_DIR/tools/play.py"

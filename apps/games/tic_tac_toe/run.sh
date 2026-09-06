#!/usr/bin/env bash
# apps/games/tic_tac_toe/run.sh
#
# Installs pygame if missing, and launches the interactive pygame match.
# Only trains when there's a real reason to: no weights exist yet, or
# you explicitly ask for it. Run from anywhere; paths below are resolved
# relative to this script and the repo root.
#
#   ./apps/games/tic_tac_toe/run.sh            # first time: trains, then plays.
#                                               #   after that: skips straight to
#                                               #   the game, since weights already
#                                               #   exist — training is a one-time
#                                               #   ~10min cost, not a per-launch one.
#   ./apps/games/tic_tac_toe/run.sh --retrain  # keep existing weights, but run
#                                               #   another 12000 epochs on top of
#                                               #   them (e.g. to push accuracy a
#                                               #   bit further)
#   ./apps/games/tic_tac_toe/run.sh --reset    # wipe weights and retrain from
#                                               #   scratch (fresh random init)
#   ./apps/games/tic_tac_toe/run.sh --play     # just launch the game, no training
#                                               #   checks at all (fails if no
#                                               #   weights exist yet)
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "== checking pygame =="
python3 -c "import pygame" 2>/dev/null || pip install pygame

WEIGHTS_DIR="cache/apps/games/tic_tac_toe/train.tl/weights"
MODE="${1:-}"

if [[ "$MODE" == "--play" ]]; then
    if [[ ! -d "$WEIGHTS_DIR" ]]; then
        echo "No trained weights found at $WEIGHTS_DIR — run without --play first."
        exit 1
    fi
    echo "== launching game (skipping all training checks) =="
    exec python3 "$APP_DIR/tools/play.py"
fi

if [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
    echo "== generating training data (minimax-labeled board states) =="
    python3 "$APP_DIR/tools/generate_data.py"
fi

WEIGHTS_EXIST=0
[[ -d "$WEIGHTS_DIR" ]] && WEIGHTS_EXIST=1

if [[ "$MODE" == "--reset" ]]; then
    echo "== initializing fresh random weights =="
    python3 "$APP_DIR/tools/init_weights.py"
    echo "== training (single run, 12000 epochs — this is the slow part, ~10min on a GPU) =="
    python3 tensorlang.py "apps/games/tic_tac_toe/train.tl"
elif [[ "$MODE" == "--retrain" ]]; then
    if [[ "$WEIGHTS_EXIST" -eq 0 ]]; then
        echo "No existing weights to retrain from — running full init + train instead."
        python3 "$APP_DIR/tools/init_weights.py"
    fi
    echo "== training another 12000 epochs on top of existing weights =="
    python3 tensorlang.py "apps/games/tic_tac_toe/train.tl"
elif [[ "$WEIGHTS_EXIST" -eq 1 ]]; then
    echo "== trained weights already exist, skipping training (use --retrain or --reset to train again) =="
else
    echo "== no trained weights found — training for the first time (single run, 12000 epochs, ~10min on a GPU) =="
    python3 "$APP_DIR/tools/init_weights.py"
    python3 tensorlang.py "apps/games/tic_tac_toe/train.tl"
fi

echo "== launching game =="
exec python3 "$APP_DIR/tools/play.py"
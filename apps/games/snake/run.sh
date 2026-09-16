#!/usr/bin/env bash
# apps/games/snake/run.sh
#
# Installs pygame if missing, generates BFS/flood-fill self-play
# training data and fresh weights if they don't exist yet, and launches
# the interactive pygame game (human play, single-AI autoplay, or a
# multi-agent arena — see tools/play.py). Training itself is NOT run
# automatically (same reasoning as connect_four's run.sh: a network
# needs an explicit --promote step before infer.tl will ever use it),
# so the game is fully playable from the very first run either way —
# agent.py falls back to the heuristic solver until you promote a
# trained network. See NOTES.md for full status.
#
#   ./apps/games/snake/run.sh              # generate data/weights if
#                                           #   missing, then play
#                                           #   (heuristic solver, until
#                                           #   you've promoted a net)
#   ./apps/games/snake/run.sh --gen-data    # (re)generate self-play
#                                           #   training data only
#   ./apps/games/snake/run.sh --reset       # fresh random weight init
#   ./apps/games/snake/run.sh --train       # train.tl on the current
#                                           #   data/weights (the slow,
#                                           #   GPU-bound part)
#   ./apps/games/snake/run.sh --promote     # promote cache/ weights to
#                                           #   the tracked production
#                                           #   location infer.tl reads
#   ./apps/games/snake/run.sh --rollback    # swap back to the previous
#                                           #   promoted weights
#   ./apps/games/snake/run.sh --play        # just launch the game, no
#                                           #   checks at all
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"

case "$MODE" in
    --play)
        exec python3 "$APP_DIR/tools/play.py"
        ;;
    --gen-data)
        python3 "$APP_DIR/tools/generate_data.py"
        exit 0
        ;;
    --reset)
        python3 "$APP_DIR/tools/init_weights.py"
        exit 0
        ;;
    --train)
        if [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
            echo "== no training data found -- generating self-play data first (see tools/generate_data.py) =="
            python3 "$APP_DIR/tools/generate_data.py"
        fi
        WEIGHTS_DIR="cache/apps/games/snake/train.tl/weights"
        if [[ ! -d "$WEIGHTS_DIR" ]] || [[ -z "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
            echo "== no weights found at $WEIGHTS_DIR -- initializing fresh random weights first =="
            python3 "$APP_DIR/tools/init_weights.py"
        fi
        echo "== pre-flight type check (no GPU needed) =="
        python3 check.py "apps/games/snake/train.tl" > /dev/null
        echo "== verifying the architecture's math against a NumPy reference (no GPU needed) =="
        python3 "$APP_DIR/tools/verify_math.py"
        echo "== training (this is the slow, GPU-bound part) =="
        python3 tensorlang.py "apps/games/snake/train.tl"
        echo "== done. Review the loss, then run --promote if you're happy with it. =="
        exit 0
        ;;
    --promote)
        exec python3 "$APP_DIR/tools/promote_weights.py"
        ;;
    --rollback)
        exec python3 "$APP_DIR/tools/rollback_weights.py"
        ;;
esac

echo "== checking pygame =="
python3 -c "import pygame" 2>/dev/null || pip install pygame

if [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
    echo "== generating self-play training data (see tools/generate_data.py) =="
    python3 "$APP_DIR/tools/generate_data.py"
fi

WEIGHTS_DIR="cache/apps/games/snake/train.tl/weights"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    echo "== initializing fresh random weights =="
    python3 "$APP_DIR/tools/init_weights.py"
fi

if [[ ! -d "$APP_DIR/weights" ]] || [[ -z "$(ls -A "$APP_DIR/weights" 2>/dev/null)" ]]; then
    echo "== no promoted network yet -- the game will use the BFS/flood-fill heuristic solver until you run --train then --promote =="
fi

echo "== launching game =="
exec python3 "$APP_DIR/tools/play.py"

#!/usr/bin/env bash
# apps/games/connect_four/run.sh
#
# Installs pygame if missing, generates self-play training data and
# fresh weights if they don't exist yet, and launches the interactive
# pygame match. Training itself is NOT run automatically (unlike
# tic_tac_toe) because a network needs an explicit --promote step
# before infer.tl will ever use it — see promote_weights.py's docstring
# for why. Until you promote a trained network, the game plays against
# the plain-Python alpha-beta solver instead (agent.py's fallback), so
# it's fully playable from the very first run either way.
#
#   ./apps/games/connect_four/run.sh                 # generate data/weights
#                                                      #   if missing, then play
#                                                      #   (vs. the solver, until
#                                                      #   you've promoted a net)
#   ./apps/games/connect_four/run.sh --gen-data       # (re)generate self-play
#                                                      #   training data only
#   ./apps/games/connect_four/run.sh --reset          # fresh random weight init
#   ./apps/games/connect_four/run.sh --train          # train.tl on the current
#                                                      #   data/weights (the slow,
#                                                      #   GPU-bound part)
#   ./apps/games/connect_four/run.sh --train \\
#       --num-positions 50000 --depth 8 --time-limit 0.15
#                                                      # regenerate a bigger/
#                                                      #   deeper self-play
#                                                      #   dataset first (any of
#                                                      #   the 3 flags forces
#                                                      #   regeneration), keep
#                                                      #   train.tl's declared
#                                                      #   shape in sync with
#                                                      #   whatever N results
#                                                      #   (tools/sync_train_shape.py),
#                                                      #   then train
#   ./apps/games/connect_four/run.sh --promote        # promote cache/ weights to
#                                                      #   the tracked production
#                                                      #   location infer.tl reads
#   ./apps/games/connect_four/run.sh --rollback       # swap back to the previous
#                                                      #   promoted weights
#   ./apps/games/connect_four/run.sh --play           # just launch the game, no
#                                                      #   checks at all
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
    shift  # drop "--train", leaving any --num-positions/--depth/--time-limit
    GEN_ARGS=()
    FORCE_REGEN=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --num-positions|--depth|--time-limit)
                GEN_ARGS+=("$1" "$2")
                FORCE_REGEN=1
                shift 2
                ;;
            *)
                echo "ERROR: unrecognized --train argument: $1"
                echo "Expected one or more of: --num-positions N --depth D --time-limit T"
                exit 1
                ;;
        esac
    done

    WEIGHTS_DIR="cache/apps/games/connect_four/train.tl/weights"
    if [[ ! -d "$WEIGHTS_DIR" ]] || [[ -z "$(ls -A "$WEIGHTS_DIR" 2>/dev/null)" ]]; then
        echo "== no weights found at $WEIGHTS_DIR — initializing fresh random weights first =="
        python3 "$APP_DIR/tools/init_weights.py"
    fi

    if [[ "$FORCE_REGEN" -eq 1 ]]; then
        echo "== regenerating self-play training data: ${GEN_ARGS[*]} =="
        python3 "$APP_DIR/tools/generate_data.py" "${GEN_ARGS[@]}"
        echo "== syncing train.tl's declared dataset size to match =="
        python3 "$APP_DIR/tools/sync_train_shape.py"
    elif [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
        echo "== no training data found — generating self-play data first (see generate_data.py) =="
        python3 "$APP_DIR/tools/generate_data.py"
        python3 "$APP_DIR/tools/sync_train_shape.py"
    fi

    echo "== pre-flight type/shape check (no GPU needed — catches shape mismatches before the slow part) =="
    python3 check.py "apps/games/connect_four/train.tl" > /dev/null
    python3 "$APP_DIR/tools/verify_dispatch.py" "apps/games/connect_four/train.tl" > /dev/null

    echo "== training (this is the slow, GPU-bound part) =="
    python3 tensorlang.py "apps/games/connect_four/train.tl"
    echo "== done. Review the loss, then run --promote if you're happy with it. =="
    exit 0
    ;;
  --promote)
    python3 "$APP_DIR/tools/promote_weights.py"
    exit 0
    ;;
  --rollback)
    python3 "$APP_DIR/tools/rollback_weights.py"
    exit 0
    ;;
esac

echo "== checking pygame =="
python3 -c "import pygame" 2>/dev/null || pip install pygame

if [[ ! -f "$APP_DIR/data/boards.npy" ]]; then
    echo "== generating self-play training data (this can take a while — see generate_data.py) =="
    python3 "$APP_DIR/tools/generate_data.py"
fi

WEIGHTS_DIR="cache/apps/games/connect_four/train.tl/weights"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    echo "== initializing fresh random weights =="
    python3 "$APP_DIR/tools/init_weights.py"
fi

if [[ ! -d "$APP_DIR/weights" ]] || [[ -z "$(ls -A "$APP_DIR/weights" 2>/dev/null)" ]]; then
    echo "== no promoted network yet — the game will use the alpha-beta solver fallback until you run --train then --promote =="
fi

echo "== launching game =="
exec python3 "$APP_DIR/tools/play.py"

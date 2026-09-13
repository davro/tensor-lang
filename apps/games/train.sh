#!/usr/bin/env bash
# apps/games/train.sh
#
# Trains every game's network in one go, non-interactively (none of
# these launch the pygame UI afterward — see each game's own run.sh for
# that). Each game's run.sh already knows how to init weights and
# generate training data if missing, so this script just calls the
# right training-only mode for each:
#
#   - tic_tac_toe: `--train-only` (exhaustively-enumerated data, single
#     ~10min run on a GPU; use tic_tac_toe/run.sh --retrain/--reset for
#     the interactive equivalents of "train more"/"start over")
#   - 2048: `--train` (never launches the game; output is scratch under
#     cache/ either way — see 2048/run.sh --promote)
#   - connect_four: `--train`, with a bigger/deeper self-play dataset
#     than the small one shipped by default (3000 positions, depth=6) —
#     any of --num-positions/--depth/--time-limit forces regeneration
#     and keeps train.tl's declared shape in sync automatically (see
#     tools/sync_train_shape.py). The values below (50000/8/0.15) trade
#     roughly 110 minutes of pure-CPU data generation for meaningfully
#     better tactical play than the shipped default — see
#     connect_four/NOTES.md's "Improving the trained network's actual
#     play quality" section for the reasoning. Adjust to taste.
#
# None of these are promoted to production automatically — review each
# game's loss/output, then run that game's own `--promote` (2048,
# connect_four) or just play (tic_tac_toe writes weights train.tl
# reads from directly, no separate promotion step).
#
# Continues past a failing game rather than aborting the whole batch —
# see the summary printed at the end for what actually succeeded.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a RESULTS=()

run_step() {
    local label="$1"
    shift
    echo ""
    echo "################################################################"
    echo "## $label"
    echo "################################################################"
    if "$@"; then
        RESULTS+=("OK    $label")
    else
        local code=$?
        RESULTS+=("FAILED $label")
        echo "!! $label failed (exit $code) — continuing with the next game !!"
    fi
}

run_step "tic_tac_toe" \
    "$SCRIPT_DIR/tic_tac_toe/run.sh" --train-only

run_step "2048" \
    "$SCRIPT_DIR/2048/run.sh" --train

run_step "connect_four" \
    "$SCRIPT_DIR/connect_four/run.sh" --train \
        --num-positions 50000 --depth 8 --time-limit 0.15

echo ""
echo "################################################################"
echo "## Summary"
echo "################################################################"
for line in "${RESULTS[@]}"; do
    echo "  $line"
done
echo ""
echo "Nothing above is promoted to production automatically:"
echo "  ./apps/games/2048/run.sh --promote"
echo "  ./apps/games/connect_four/run.sh --promote"
echo "(tic_tac_toe has no separate promotion step — its own train.tl"
echo "output is what tools/infer.tl reads directly.)"

for line in "${RESULTS[@]}"; do
    [[ "$line" == FAILED* ]] && exit 1
done
exit 0

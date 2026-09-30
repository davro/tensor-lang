#!/usr/bin/env bash
# apps/games/train.sh
#
# Trains every game's network in one go, non-interactively (none of
# these launch the pygame UI afterward — see each game's own
# `tensorlang.py --app games/<name>` for that). Each game's [lifecycle]
# already knows how to init weights and generate training data if
# missing, so this script just calls the right training-only mode for
# each:
#
#   - tic_tac_toe: `--step train_only` (exhaustively-enumerated data,
#     single ~10min run on a GPU; use `--step retrain`/`--step
#     reset` for the interactive equivalents of "train
#     more"/"start over")
#   - 2048: `--step train` (never launches the game; output is scratch
#     under cache/ either way — see `--step promote`)
#   - connect_four: gen_data + sync_shape + train as three explicit
#     steps, with a bigger/deeper self-play dataset than the small one
#     shipped by default (3000 positions, depth=6) — --step train alone
#     only regenerates data if data/boards.npy is missing, and by now it
#     usually isn't, so a forced regen with different hyperparameters
#     has to be spelled out as its own steps rather than folded into
#     [lifecycle]'s "only if missing" pipeline shape. The values below
#     (50000/8/0.15) trade roughly 110 minutes of pure-CPU data
#     generation for meaningfully better tactical play than the shipped
#     default — see connect_four/NOTES.md's "Improving the trained
#     network's actual play quality" section for the reasoning. Adjust
#     to taste.
#
# None of these are promoted to production automatically — review each
# game's loss/output, then run that game's own `--step promote` (2048,
# connect_four) or just play (tic_tac_toe writes weights train.tl
# reads from directly, no separate promotion step).
#
# Continues past a failing game rather than aborting the whole batch —
# see the summary printed at the end for what actually succeeded.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

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
    python3 tensorlang.py --app games/tic_tac_toe --step train_only

run_step "2048" \
    python3 tensorlang.py --app games/2048 --step train

run_step "connect_four (gen_data)" \
    python3 tensorlang.py --app games/connect_four --step gen_data \
        --app-args --num-positions 50000 --depth 8 --time-limit 0.15
run_step "connect_four (sync_shape)" \
    python3 tensorlang.py --app games/connect_four --step sync_shape
run_step "connect_four (train)" \
    python3 tensorlang.py --app games/connect_four --step train

echo ""
echo "################################################################"
echo "## Summary"
echo "################################################################"
for line in "${RESULTS[@]}"; do
    echo "  $line"
done
echo ""
echo "Nothing above is promoted to production automatically:"
echo "  python3 tensorlang.py --app games/2048 --step promote"
echo "  python3 tensorlang.py --app games/connect_four --step promote"
echo "(tic_tac_toe has no separate promotion step — its own train.tl"
echo "output is what tools/infer.tl reads directly.)"

for line in "${RESULTS[@]}"; do
    [[ "$line" == FAILED* ]] && exit 1
done
exit 0


#!/usr/bin/env bash
# apps/examples/decision_boundary/run.sh
#
# Installs pygame if missing, bootstraps state on first run, trains all
# chunks, then launches the viewer. Run from anywhere; paths below are
# resolved relative to this script and the repo root.
#
#   ./apps/examples/decision_boundary/run.sh          # normal run
#   ./apps/examples/decision_boundary/run.sh --reset   # wipe weights and start fresh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "== checking pygame =="
python3 -c "import pygame" 2>/dev/null || pip install pygame

WEIGHTS_DIR="cache/apps/examples/decision_boundary/main.tl/weights"
if [[ "${1:-}" == "--reset" || ! -d "$WEIGHTS_DIR" ]]; then
    echo "== initializing state (fresh weights + grid) =="
    python3 "$APP_DIR/tools/init_state.py"
else
    echo "== existing weights found, resuming training (use --reset to start over) =="
fi

echo "== training =="
python3 "$APP_DIR/tools/train_and_snapshot.py"

echo "== launching viewer =="
python3 "$APP_DIR/tools/viewer.py"

#!/usr/bin/env bash
# Run from the repo root:  bash /path/to/new_tests/install.sh /path/to/new_tests
set -euo pipefail
src="${1:-.}"
cp "$src"/*.tl "$src"/*.npy tests/
echo "copied; now run: python3 tensorlang.py --test --filter elif_scalar; ... or the full suite"

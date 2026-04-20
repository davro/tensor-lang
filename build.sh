#!/usr/bin/env bash
# =============================================================================
# build.sh — TensorLang build, test, and development runner
#
# Usage:
#   source build.sh                         # full test suite (default)
#   source build.sh --install               # install/upgrade all dependencies
#   source build.sh --lint                  # ruff lint check
#   source build.sh --data                  # run data pipeline scripts
#   source build.sh --test                  # run full test suite explicitly
#   source build.sh --test --filter NAME    # run tests matching NAME
#   source build.sh --debug FILE.tl         # compile a single .tl file with debug
#   source build.sh FILE.tl                 # compile a single .tl file
#
# Options (can be combined):
#   --install       Install/upgrade Python dependencies
#   --lint          Run ruff linter over tensorlang/
#   --data          Run data pipeline (data/bitcoin_daily.py)
#   --test          Run test suite (default when no file given)
#   --filter NAME   Filter tests by name pattern (requires --test)
#   --debug         Enable debug output
#   --no-cache      Skip cache-layers flag (faster for single-file runs)
#   --clean         Delete the cache/ directory before running
#   --help          Show this help message
# =============================================================================

set -euo pipefail

# ─────────────────────────── defaults ────────────────────────────────────────
ARG=""
DEBUG=""
FILTER=""
INSTALL=0
LINT=0
DATA=0
RUN_TEST=0
NO_CACHE=0
CLEAN=0

# ─────────────────────────── parse arguments ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            sed -n '2,20p' "$0"   # print the header comment block
            exit 0
            ;;
        --install)
            INSTALL=1
            shift
            ;;
        --lint)
            LINT=1
            shift
            ;;
        --data)
            DATA=1
            shift
            ;;
        --test)
            RUN_TEST=1
            shift
            ;;
        --filter)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                FILTER="$2"
                shift 2
            else
                echo "Error: --filter requires a pattern argument" >&2
                exit 1
            fi
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --no-cache)
            NO_CACHE=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Run 'source build.sh --help' for usage." >&2
            exit 1
            ;;
        *)
            if [[ -z "$ARG" ]]; then
                ARG="$1"
                shift
            else
                echo "Error: only one positional argument allowed." >&2
                exit 1
            fi
            ;;
    esac
done

# ─────────────────────────── virtualenv ──────────────────────────────────────
python3 -m venv python-env
source python-env/bin/activate

# ─────────────────────────── install ─────────────────────────────────────────
if (( INSTALL )); then
    echo "=== Installing dependencies ==="
    pip install --upgrade pip

    pip install ruff
    pip install lark
    pip install pycuda
    pip install numpy
    pip install tqdm
    pip install pandas
    pip install yfinance
    pip install tomli

    pip freeze > requirements.txt
    echo "=== requirements.txt updated ==="
    exit 0
fi

# ─────────────────────────── lint ────────────────────────────────────────────
if (( LINT )); then
    echo "=== Linting tensorlang/ ==="
    ruff check tensorlang/
    exit 0
fi

# ─────────────────────────── data pipeline ───────────────────────────────────
if (( DATA )); then
    echo "=== Running data pipeline ==="
    python3 data/bitcoin_daily.py
    exit 0
fi

# ─────────────────────────── cache clean ─────────────────────────────────────
# --clean flag: always wipe cache before running
# Default full-suite run: always wipe cache (prevents stale .npy poisoning tests)
if (( CLEAN )); then
    echo "=== Cleaning cache/ ==="
    rm -rf cache/
fi

# ─────────────────────────── single file ─────────────────────────────────────
if [[ -n "$ARG" ]]; then
    if [[ ! -f "$ARG" ]]; then
        echo "Error: file not found: $ARG" >&2
        exit 1
    fi

    CACHE_FLAG=""
    if (( ! NO_CACHE )); then
        CACHE_FLAG="--cache-layers"
    fi

    echo "=== Compiling: $ARG ==="
    python3 tensorlang.py $CACHE_FLAG $DEBUG "$ARG"
    exit 0
fi

# ─────────────────────────── test suite ──────────────────────────────────────
# Default mode when no file or special flag is given, or --test is explicit.
# Always wipe cache before the full suite to prevent stale .npy files from
# previous runs causing false passes or wrong @EXPECTED comparisons.

if (( RUN_TEST )) || [[ -z "$ARG" ]]; then
    # Wipe cache unless --no-cache was passed (useful for quick re-runs)
    if (( ! NO_CACHE )); then
        if [[ -d "cache/" ]]; then
            echo "=== Clearing cache/ before test run ==="
            rm -rf cache/
        fi
    fi

    FILTER_FLAG=""
    if [[ -n "$FILTER" ]]; then
        FILTER_FLAG="--filter $FILTER"
    fi

    echo "=== Running test suite ==="
    python3 tensorlang.py --cache-layers --verify-tensors --test $DEBUG $FILTER_FLAG
fi
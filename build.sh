#!/usr/bin/env bash
# ------------------------------------------------------------
# run_tensorlang.sh
#
# Usage:
#   ./run_tensorlang.sh [ARG] [--install] [--data [path]]
#
#   * If ARG is given  →  python tensorlang.py --debug ARG [...]
#   * If ARG is missing → python3 tensorlang.py --cache-layers --verify-tensors --test [...]
#
#   --install   : run the install step (pip install -e .)
#   --data [p]  : pass --data p (or just --data if no path)
# ------------------------------------------------------------

#set -euo pipefail   # safer scripting

# ----------------------- defaults ---------------------------
DEBUG=""
TEST=""
INSTALL=0
LINT=0
DATA=0
DATA_FLAG=""
ARG=""

# --------------------- parse options ------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --test)
            TEST="--test"
            shift
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
            # optional value: --data path  or just --data
            #if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
            #    DATA_FLAG="--data $2"
            #    DATA=1
            #    shift 2
            #else
            #    DATA=1
            #    shift
            #fi
            ;;
        -*)
            echo "Unknown option: $1" >&2
            #exit 1
            exit
            ;;
        *)
            # first non-option is the positional ARG
            if [[ -z "$ARG" ]]; then
                ARG="$1"
                shift
            else
                echo "Only one positional argument allowed." >&2
                #exit 1
            fi
            ;;
    esac
done

# Python virtual environment setup
python3 -m venv python-env
source python-env/bin/activate


# ----------------  build the python command -----------------
if [[ -n "$ARG" ]]; then
    `python tensorlang.py --cache-layers --verify-tensors $DEBUG $TEST "$ARG"`
elif (( INSTALL )); then
    # System install python venv required for Ubuntu.
    #sudo apt install python3.12-venv

    # Ollama
    #pip install ollama

    # Requirements used for development andbuilding requirements
    pip install yfinance
    pip install ruff
    pip install lark
    pip install --upgrade lark
    pip install pycuda
    pip install numpy
    pip install tqdm
    pip install tomli
    pip freeze > requirements.txt

    # Install required packages
    #pip install -r requirements.txt

elif (( LINT )); then
    # Code check linting
    ruff check tensorlang/

elif (( DATA )); then

    #find . -name "*.pyc" -delete
    #find . -name "__pycache__" -delete

    # echo "===== VERIFY NPZ ====="
    # python -c "
    # import numpy as np
    # with np.load('data/bitcoin_daily.npz') as f:
    #     data = f['data']
    #     print('Last 10 Opens:', data[-10:, 0])
    # "

    #"rm -f data/bitcoin_daily.npz"
    python data/bitcoin_daily.py
    #python tensorlang.py --test data/bitcoin.tl
    #CMD=(
    #    "python data/bitcoin_daily.py"
    #    "python tensorlang.py --test data/bitcoin.tl"
    #)

else
    if [[ -d "cache/" && ! -L "cache/" ]]; then
        rm -rf cache/
    fi

    python3 tensorlang.py --cache-layers --verify-tensors --test
fi

# ----------------------- execute ----------------------------
#echo "Executing: ${CMD[*]}"
#"${CMD[@]}"

# Python virtual environment deactivate
#deactivate

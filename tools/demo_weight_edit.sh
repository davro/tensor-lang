#!/bin/bash
set -e

WEIGHTS="${1:-build/gpt2}"
OUT="${2:-build/gpt2_edited}"
PROMPT="${3:-Paris is the capital of}"
SEARCH="${4:- the}"
REPLACE="${5:- England}"
LAYER="${6:-7}"
ALPHA="${7:-4.0}"

echo "=== Weight-space edit demo ==="
echo "weights : $WEIGHTS"
echo "out     : $OUT"
echo "prompt  : $PROMPT"
echo "search  : $SEARCH"
echo "replace : $REPLACE"
echo "layer   : $LAYER  alpha=$ALPHA"
echo

rm -rf "$OUT"
python3 tools/weight_edit.py \
  --weights "$WEIGHTS" --out "$OUT" \
  --prompt "$PROMPT" --search "$SEARCH" --replace "$REPLACE" \
  --layer "$LAYER" --alpha "$ALPHA" --mode bias

echo
echo "=== Compile + run in TensorLang ==="
python3 tools/make_gpt2_demo.py --weights "$OUT" --prompt "$PROMPT" --out build/gpt2_forward.tl
python3 tensorlang.py build/gpt2_forward.tl

echo
echo "=== Numerical agreement check ==="
python3 tools/compare_gpt2_logits.py

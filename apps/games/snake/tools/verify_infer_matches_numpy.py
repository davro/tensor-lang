#!/usr/bin/env python3
"""
Confirms that agent.run_inference_fast (the in-process NumPy mirror
choose_ai_move actually uses during gameplay — see agent.py's module
comment for why) produces the same output as agent.run_inference (the
real infer.tl program, actually compiled and run through TensorLang).

This needs a promoted network AND a working TensorLang/CUDA setup —
run it once on real GPU hardware right after any tools/promote_weights.py,
before trusting the fast path for a real game:

    python3 apps/games/snake/tools/verify_infer_matches_numpy.py

Why this check exists at all, rather than just trusting the two agree:
verify_math.py already proves the underlying MATH (matmul/relu/softmax
backward pass) is internally consistent, but that's a from-scratch NumPy
derivation, not a comparison against infer.tl's actual output — a typo
in forward_numpy (wrong weight file, a transposed matmul, a missed
relu) could still make the two diverge silently. Neither this repo's
sandbox nor the one that wrote this file has a GPU, so this has NOT
been run yet — see NOTES.md.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import engine  # noqa: E402

NUM_BOARDS = 12
TOLERANCE = 1e-4  # softmax outputs; TensorLang's CUDA kernels and NumPy
                   # won't be bit-identical (different reduction order,
                   # fp32 accumulation) but should agree to several
                   # decimal places for the same weights and inputs.


def random_board(rng):
    snake = engine.new_snake(rng)
    for _ in range(rng.randint(0, 40)):
        food = engine.place_food(rng, snake.occupied())
        d = agent.heuristic_choose_move(snake.body, snake.direction, food)
        if d is None:
            break
        snake = engine.step_snake(snake, d, food)
        if not snake.alive:
            break
    food = engine.place_food(rng, snake.occupied())
    return snake.body, food


def main():
    try:
        agent.load_promoted_weights()
    except FileNotFoundError as e:
        print(f"No promoted weights found ({e}) — run ./run.sh --train then --promote first.")
        sys.exit(1)

    rng = random.Random(123)
    worst = 0.0
    failures = 0
    for i in range(NUM_BOARDS):
        body, food = random_board(rng)
        fast = agent.run_inference_fast(body, food)
        try:
            real = agent.run_inference(body, food)
        except RuntimeError as e:
            print(f"board {i}: infer.tl failed to run at all: {e}")
            failures += 1
            continue
        diff = np.abs(fast - real)
        max_diff = float(diff.max())
        worst = max(worst, max_diff)
        status = "OK" if max_diff < TOLERANCE else "MISMATCH"
        print(f"board {i}: max |fast - real| = {max_diff:.2e}  [{status}]  "
              f"fast_argmax={agent.DIRECTIONS[int(fast.argmax())]} "
              f"real_argmax={agent.DIRECTIONS[int(real.argmax())]}")
        if max_diff >= TOLERANCE:
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures}/{NUM_BOARDS} boards exceeded tolerance {TOLERANCE:.0e} "
              f"(worst {worst:.2e}) or errored — do not trust run_inference_fast until this "
              f"is resolved (check forward_numpy against train.tl/infer.tl for a transcription "
              f"error).")
        sys.exit(1)
    print(f"OK — all {NUM_BOARDS} boards agreed within {TOLERANCE:.0e} (worst {worst:.2e}). "
          f"The fast NumPy path is a faithful mirror of infer.tl.")


if __name__ == "__main__":
    main()

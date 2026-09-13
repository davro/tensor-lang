# apps/games/connect_four/ — status and how to run it

**Status (2026-09-12): engine, solver, data pipeline, and pygame frontend are
built and heavily tested; train.tl/infer.tl are shape- and math-verified
without a GPU but have NEVER been run on real CUDA hardware.** This app was
built in a sandbox with no `nvcc`, no `pycuda`, and no GPU at all — see
"What was and wasn't verified" below before trusting a promoted network.

## What this is

A policy+value MLP for Connect Four, following the same overall shape as
`tic_tac_toe` (TensorLang only runs the neural net; game rules live in
Python) and reusing `2048`'s promote/rollback pattern for taking a trained
network into production. Unlike tic_tac_toe, this network has two output
heads instead of one:

```
input (42) -- one cell per board square, mover's perspective:
              +1 mover's disc, -1 opponent's disc, 0 empty
  |
shared trunk: matmul/add/relu (42->128), matmul/add/relu (128->64)
  |                                              |
policy head (64->7, softmax)          value head (64->1, tanh)
  |                                              |
policy_loss = mse_loss(...)            value_loss = mse_loss(...)
                    \                    /
                     loss = add(policy_loss, value_loss)
```

Also unlike tic_tac_toe (4520 total reachable states — exhaustively
enumerable and exactly solvable), Connect Four has ~4.5 trillion legal
positions. `tools/generate_data.py` self-plays a fixed number of games with
a depth-limited alpha-beta solver instead of enumerating/solving exactly,
and labels each position two ways: a policy target (the solver's clean best
move for that exact position) and a value target (the actual outcome of
that self-play game, backfilled after the fact — standard self-play-style
value labeling, not a static heuristic score).

## The Python engine (tools/engine_bitboard.py, tools/engine_solver.py)

A complete, independently-useful Connect Four engine, vendored from a
sibling non-TensorLang project (bitboard.py, solver.py) into this app's
tools/ directory:

- `engine_bitboard.py`: Fhourstones-style bitboard board representation
  (49 bits: 7 columns x 7 rows including a sentinel row), O(1) move
  legality/win checks via bit tricks, plus a `Game` wrapper that also
  tracks per-player disc ownership for display.
- `engine_solver.py`: two solvers sharing those bitboard primitives —
  `ExactSolver` (negamax + alpha-beta + transposition table, exact but
  only practical for midgame/endgame analysis in pure Python — see its
  `SolverTimeout` docstring for why a full 42-ply solve from an early
  position isn't realistic here), and `HeuristicSolver` (depth-limited
  alpha-beta + a positional evaluation function, fast enough for
  real-time play and what `agent.py` actually falls back to).

**A real bug was found and fixed while building this**: `Game.play()`'s
landing-row calculation (`bin(mask >> (col*7)).count("1")`) didn't mask off
bits belonging to OTHER columns before counting, so once some other column
(especially a higher-indexed one) already had discs, the shift could pull
those bits into range and inflate the count — discs could visually land on
the wrong row. This didn't affect the core solver (`Board.play()`'s own
mask/position bit tricks are unaffected and were separately validated
first), but it corrupted `Game.grid()` — which `generate_data.py`'s and
`agent.py`'s `encode_board()` both read — so it would have silently trained
on wrong board encodings and corrupted the pygame display. Caught by a
`winning_cells()` display test finding a "diagonal win" in a hand-built
horizontal-win scenario; verified fixed with a 500-game stress test that
cross-checks `grid()` after every single move against an independent
Python-side column-height tally. Fixed in both this app's vendored copy and
the sibling project's original.

**A second real bug was found and fixed in `train.tl` itself** (not the
compiler): `mult(lr, bv_grad)` — the value head's single-unit bias update —
paired a `(1,1)` scalar with a `(1,)` vector. `compiler.py`'s dispatcher
routes any 2D-vs-1D operand pair into `KernelGenerator.binary_broadcast`,
which unconditionally does `rows, cols = output_shape`; here the correct
output is 1D `(1,)`, so it crashed unpacking a 1-tuple into two variables.
This exact combination — a true `(1,1)` scalar against a *width-1* 1D
vector — had never been exercised by `tic_tac_toe`/`2048`, whose biases are
always width > 1. Fixed the same way those apps sidestep the *documented*
version of this class of bug (`(1,1)` vs a *wider* 1D vector): a matching
same-shape `lr_bv` routes the update through the simple, proven
`shape1==shape2` elementwise path instead.

**Both bugs were caught by actually simulating the compiler, not just
`check.py`.** `check.py`'s type checker validates shapes are *compatible*
but never simulates `compiler.py`'s own branch-by-branch kernel dispatch —
so it happily approved both the malformed literal array (right shape,
wrong element count) and the `bv` mult (right output shape, wrong dispatch
branch for that shape combination). `tools/verify_dispatch.py` closes that
gap: it walks every `add`/`minus`/`mult`/`div` call in a `.tl` file
(including inside `for` loops, which `check.py`'s own env dump also
includes but which naive AST traversal misses — the loop body is nested
under the `for` node, not flattened into the top-level statement list),
replicates `compiler.py::_kernel_for_expr`'s exact shape-based branching,
and flags anything that would hit `binary_broadcast` without a genuinely
2D output, or that lands on `binary_general_broadcast` — a path this app's
learning-rate updates avoid entirely except for the `(1,1)`-scalar-times-
weight-matrix case, which is the exact same pattern `tic_tac_toe`'s
already-hardware-verified `mult(lr, w1_grad)` uses. Run it on any `.tl`
file before trusting `check.py` alone:
```bash
python3 apps/games/connect_four/tools/verify_dispatch.py apps/games/connect_four/train.tl
python3 apps/games/connect_four/tools/verify_dispatch.py apps/games/connect_four/infer.tl
```
Both currently report zero problems, with every remaining op traced to a
route already proven either by `tic_tac_toe`/`2048`'s hardware-verified
runs or by being a trivial same-shape elementwise call.



**Verified, with no GPU needed:**
- The Python engine: hundreds of targeted + randomized tests (win
  detection in all 4 directions, tactical must-win/must-block behavior,
  the landing-row bug above, 500-game grid-consistency stress test).
- `train.tl` and `infer.tl` parse and TYPE CHECK against the real
  TensorLang compiler (`check.py`, which needs `lark`+`numpy` only) —
  every declared shape, every inferred `_grad` shape, resolves correctly.
- Both files also pass `tools/verify_dispatch.py`, which goes one level
  deeper than `check.py` and replicates `compiler.py`'s actual
  branch-by-branch kernel dispatch logic for every `add`/`minus`/`mult`/
  `div` call (this is what caught the `bv` bug above — `check.py` alone
  did not).
- The architecture's MATH: `tools/verify_math.py` hand-derives the
  forward+backward pass in NumPy and finite-difference-checks every
  gradient (max relative error ~1e-6/1e-7) — this validates the DESIGN
  (is `loss = add(policy_loss, value_loss)` a correct thing to
  `backward()`? do both heads receive correct gradients?), independent
  of whether TensorLang's CUDA kernels implement each op correctly.
- `tools/generate_data.py` was actually run end-to-end (pure Python, no
  GPU needed) — the shipped `data/*.npy` is real, not a placeholder: 3000
  positions from 102 self-play games (depth=6, time_limit=0.08s/move,
  ~200s total). See its docstring for the exact settings and stats.
- `agent.py`'s fallback path was exercised for real: with no `pycuda`
  installed, `infer.tl` genuinely fails to run, and `choose_move()`
  correctly catches that and falls back to `HeuristicSolver` — confirmed
  it still picks a sound move (the center-column opening).
- `tools/play.py`: a headless smoke test (SDL dummy driver) exercises the
  start screen, human clicks, AI-fallback replies, win detection/ring
  highlighting, and restart — no crashes.
- `promote_weights.py`/`rollback_weights.py`: tested end-to-end with
  placeholder weight files — promote copies+backs-up correctly, rollback
  swaps correctly, running rollback twice returns to the start state.
- `tools/stats.py`: tested end-to-end through `play.py` — deterministic
  scripted games for both winners, confirmed no double-counting across
  repeated frames of a finished game, confirmed persistence across
  separate process instances.

**NOT verified — needs real GPU hardware:**
- That TensorLang's actual CUDA kernels for `matmul`/`add`/`relu`/`tanh`/
  `softmax`/`mse_loss` numerically agree with the NumPy reference above.
  Individual ops are documented elsewhere in this repo as hardware-verified
  (HANDOVER.md, via tic_tac_toe/decision_boundary), but the specific
  COMBINATION here — two `mse_loss` outputs `add()`ed together and
  `backward()`'d as one, feeding two independent heads off a shared
  trunk — has never actually been compiled or run.
- Whether 8000 epochs at lr=8.0 is remotely a good setting. This is a
  starting point chosen by analogy to tic_tac_toe's (also un-tuned-by-me)
  12000-epochs/lr=16 setup, not by observing an actual loss curve.
- Whether 3000 self-play positions (vs. tic_tac_toe's exhaustive 4520
  *guaranteed-correct* states) is enough data for the network to
  meaningfully beat the solver it was trained to imitate, given Connect
  Four's vastly larger state space. Likely not enough for strong play —
  regenerate a bigger dataset (`--num-positions`) once training is known
  to work at all.

## Bug #3 (found on real GPU hardware) and the fix

Training completed successfully (`loss_final = 1.2067`, ~1234s) and
promotion worked, but `--play` crashed on the very first AI move:
`add(v_pre, bv)` inside `infer.tl` (batch size 1) hit the exact same
`binary_broadcast` / `rows, cols = output_shape` crash as bug #2, but
through `add`, not `mult`, and through a route the first version of
`verify_dispatch.py` missed.

Root cause: `type_checker.py`'s `_is_scalar_shape` helper (`len(shape)==0
or all(d==1 for d in shape)` — intended to recognize `mse_loss`'s own
`(1,1)` output as a scalar) also matches any tensor whose dims are all
1, including a perfectly ordinary `(1,1)` 2D layer output at batch size
1. Since the value head had exactly 1 output unit, `v_pre` at
`infer.tl`'s batch size of 1 is `(1,1)` — genuinely 2D, but
indistinguishable from a scalar to this heuristic — so `add(v_pre, bv)`
was silently typed as producing a 1D `(1,)` output instead of `(1,1)`,
then crashed the same `binary_broadcast` unpack. A batch=3000 training
run can never surface this (`v_pre` is `(3000,1)` there, never
all-ones), which is exactly why it passed training and promotion
cleanly and only broke at single-sample inference.

This also exposed a real gap in `verify_dispatch.py` itself: the first
version re-derived its own approximation of the type checker's
output-shape rule instead of using the real one (already sitting in
`env`, computed by the actual `type_checker`), and that approximation
happened to agree with reality for `train.tl` (batch=3000 never
produces an all-ones shape) but silently disagreed for `infer.tl`
(batch=1 does). Fixed by using `env`'s real computed shape as ground
truth everywhere — rerunning the corrected tool immediately reproduces
the exact crash that occurred on real hardware.

The actual fix: rather than patch around this one `add()` call (as bug
#2's `lr_bv` fix addressed the SGD *update* step but not this forward
pass), the value head now has **2 output units, not 1** — the same
"always width > 1" property every other layer in this file already
has, so it can never produce an all-ones shape at any batch size.
`y_value`'s second column is trained as the negation of the first (see
`tools/generate_data.py`); `tools/agent.py` averages `(col0 - col1)/2`.
`tools/verify_dispatch.py` on the updated `infer.tl` now shows
`v_biased` producing a genuine `(1, 2)` output through the same,
already-proven `binary_broadcast` path every other bias-add in this
file uses — no exotic dispatch route needed. `check.py` and
`verify_dispatch.py` both pass cleanly on the updated `train.tl` and
`infer.tl`, and the dataset/weights were regenerated to match the new
`(N, 2)` value shape.

## Win/loss tracking (`data/play_stats.json`)

`tools/stats.py` persists cumulative results across `play.py` sessions,
broken down by mode (`human_vs_ai` vs `ai_self_play`) and by which side
won (red = player 1, blue = player 2 — the pygame UI's colors, updated
from red/yellow to red/blue per request). Deliberately a separate file
from `data/meta.json`, which describes the one-time self-play
*training* dataset `generate_data.py` produced — `play_stats.json`
instead grows across as many interactive sessions as you like. Shown on
both the start screen and the in-game status bar. Tested end-to-end:
deterministic scripted wins for both sides, confirmed no double-counting
across repeated frames of a finished game (the result is recorded once,
guarded by a per-game flag cleared on reset), and confirmed persistence
across separate `App` instances (i.e. across actual process restarts).

## Suggested first real run, on a machine with a GPU + CUDA toolkit

```bash
cd tensor-lang
bash build.sh --install               # if not already done
source python-env/bin/activate
python3 apps/games/connect_four/tools/verify_math.py     # no GPU needed, run first
python3 check.py apps/games/connect_four/train.tl        # no GPU needed
python3 check.py apps/games/connect_four/infer.tl        # no GPU needed
./apps/games/connect_four/run.sh --train                 # the actual GPU run
# inspect cache/apps/games/connect_four/train.tl/{loss,policy_loss,value_loss}.npy —
# if policy_loss and value_loss are both decreasing sensibly, then:
./apps/games/connect_four/run.sh --promote
./apps/games/connect_four/run.sh --play
```

If `--train` fails or produces garbage, `run.sh --play` still works —
`agent.py` falls back to `HeuristicSolver`, so the game is never blocked
on a working network.

## Tactical safety net (`agent.py`'s `choose_move`)

A network trained on a few thousand self-play positions can visibly miss
one-move-obvious tactics — taking an immediate win, or blocking an
opponent's immediate win — especially early on with limited data.
Rather than relying on more training to *probably* fix this,
`choose_move` now checks `game.board.winning_moves()` /
`opponent_winning_moves()` (the same O(7), exact bitboard checks
`HeuristicSolver` itself already used) BEFORE consulting the network or
the solver fallback at all. If there's an immediate win, or exactly one
forced block, it's taken immediately — the network is never even
called, so this is a structural guarantee, not something training
quality affects. (If there are 2+ simultaneous winning threats to
block, no single move stops both — that position is already lost, so
this intentionally does NOT special-case it; the network/solver picks
its own least-bad losing move instead.) Shown in the status bar as
"tactical override (immediate win)" / "(forced block)". Tested against
scripted immediate-win and forced-block positions, and against a
found-by-search genuine double-threat position to confirm it correctly
declines to special-case an unavoidable loss.

This alone should eliminate most "obviously wrong" moves reported while
playing. Anything still wrong past 1-ply tactics is a real capability
gap in the network (or the solver fallback's search depth), not a
missing safety net — see below for closing that gap.

## Improving the trained network's actual play quality

The shipped defaults (3000 self-play positions, solver depth=6,
8000 epochs at lr=8.0) were chosen to be a fast, working starting
point, not a strong one — see NOTES.md's earlier "NOT verified" section.
In roughly the order of impact per unit effort:

1. **More training data, first.** Connect Four has ~4.5 trillion legal
   positions; 3000 labeled examples covers essentially none of it. This
   is almost certainly the single biggest lever:
   ```bash
   python3 apps/games/connect_four/tools/generate_data.py --num-positions 50000 --depth 8 --time-limit 0.15
   ```
   At ~0.13s/position at these settings, 50000 positions is roughly
   110 minutes of pure-CPU (no GPU needed) time — leave it running.
   Update `train.tl`'s three `Tensor[f32, (N, ...)]` declarations
   (currently `3000`) to match `N` exactly, and re-run `init_weights.py`
   (a bigger dataset with the old random init is fine, but don't mix
   old weights trained on 3000 positions with a differently-sized
   dataset without re-running `check.py`/`verify_dispatch.py` again —
   shape mismatches surface as compiler errors, not silently).
2. **Deeper solver labels.** `--depth 6 --time-limit 0.08` (the shipped
   default) is fast but shallow — deeper search means better-quality
   policy/value labels for the network to imitate, at the cost of
   slower data generation (still no GPU needed, just CPU time).
3. **Mirror-image data augmentation** (not yet implemented): Connect
   Four boards have an exact left-right mirror symmetry — flipping a
   board horizontally and reversing the policy target's 7 columns
   produces another perfectly valid, free training example. This
   roughly doubles the effective dataset size for zero additional
   solver computation. Would go in `generate_data.py`, right before
   `np.save` — for each recorded `(board_vec, policy, mover)`, also
   append `(mirrored_board_vec, mirrored_policy, mover)`.
4. **More training, or a tuned learning rate.** 8000 epochs at lr=8.0
   was chosen by analogy to `tic_tac_toe`'s (also not tuned by
   observing a real loss curve) 12000/16.0 setup. Once a training run
   completes, actually look at `cache/apps/games/connect_four/train.tl/
   {policy_loss,value_loss}.npy` — if either is still dropping fast at
   epoch 8000, more epochs helps; if `loss_final` plateaus early or
   oscillates, the learning rate is probably too high.
5. **Iterative self-play** (bigger project): once a network exists,
   `generate_data.py` could use it (via `agent.py`, blended with the
   solver) to generate a second, stronger round of training data —
   closer to how AlphaZero-style training actually works, rather than
   a single solver-labeled batch. Not implemented here.

## `cuInit failed: unknown error` (intermittent CUDA driver failures)

`agent.py` spawns `infer.tl` as a fresh subprocess for every single AI
move, each one initializing its own CUDA context from scratch via
`pycuda.autoinit`. If this fails intermittently (works for some moves,
fails for others in the same session — the fallback solver quietly
covers the failures, so the game stays playable either way), that's a
driver/environment issue outside this app's code, and outside what
could be diagnosed or fixed from this sandbox (no GPU here at all).
Things worth checking on your machine:
- `nvidia-smi` succeeding in the same shell you launch `run.sh --play`
  from, right before/during a failure.
- Whether the `--train` run's long-lived CUDA context was still
  shutting down / hadn't released the GPU when `--play`'s first rapid
  subprocess calls started.
- Any concurrent process (another training run, another game session)
  holding the GPU.
- `dmesg`/`nvidia-bug-report.sh` around the failure timestamp for a
  driver-level error.
If it's reproducible on a quiet GPU with nothing else running, that
likely points at something in how frequently/rapidly this app spawns
fresh CUDA contexts (one per move) rather than the model or code itself
— worth raising against TensorLang directly if so.

## Regenerating a bigger/better dataset

```bash
python3 apps/games/connect_four/tools/generate_data.py --num-positions 20000 --depth 8 --time-limit 0.15
```
At the measured ~0.13s/position for those (default) settings, 20000
positions is roughly 40 minutes of pure-CPU, no-GPU-needed time. Update
`train.tl`'s three `Tensor[f32, (N, ...)]` declarations to match N exactly
if you change `--num-positions`.

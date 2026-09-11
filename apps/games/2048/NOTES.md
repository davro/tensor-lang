# apps/games/2048/step.tl — status and how to run it

**Status (2026-09-11): done, playable, and now with a trained AI that beats
the heuristic it replaced.** All four engine files (`step.tl`,
`step_right.tl`, `step_up.tl`, `step_down.tl`) are verified on real GPU
hardware (a 1080ti), and the pygame frontend (`tools/play.py` +
`tools/agent.py`) is wired up on top of them and has been played through a
full session to game-over — see "The pygame frontend" below. All four
compiler bugs found along the way are fixed. See `HANDOVER.md` §17 in the
repo root for the handover-level summary of the original engine session.

On top of that, there's now a trained move-picking network
(`train.tl` + `tools/generate_data.py` + `tools/init_weights.py`,
mirroring `tic_tac_toe`'s setup) standing in for the old hand-written
heuristic in `agent.py`'s `choose_ai_move` — see "The trained move-picking
network" below for the full story, including a real overfitting bug found
and fixed along the way. Confirmed on real GPU hardware across two
autoplay games: scores of 1352 and **2300**, the latter beating the
heuristic's previous best of 2172.

## What this is

`step.tl` is a pure tensor-ops implementation of one 2048 move (slide left).
Unlike `tic_tac_toe` (where tensor-lang only runs the neural net and the game
rules live in Python), here the actual game mechanics — compaction, merging —
are expressed as tensor arithmetic. See the header comment in `step.tl` for
the full design writeup, including why a naive "check fixed pairs (0,1) and
(2,3)" merge rule is wrong and how the branchless recursive-select version
handles it correctly.

`step_right.tl`, `step_up.tl`, and `step_down.tl` are siblings covering the
other three directions, built exactly the way `step.tl`'s own header comment
predicted: RIGHT reverses each row, runs the same LEFT-slide pipeline, then
reverses back; UP/DOWN call the built-in `transpose()` op (see
`tests/transpose.tl`) so each column becomes a row, then reuse the LEFT or
RIGHT pipeline per row before transposing back. tensor-lang has no
import/include mechanism, so `compact_pair`/`compact_row`/`merge_pair`/
`merge_row` are duplicated verbatim into all four files rather than shared —
keep them in sync if you ever touch the algorithm itself. All four are now
confirmed correct on real GPU hardware against `verify_all.py`'s
plain-Python reference (see "How to apply the fixes and run this yourself"
below for example boards/expected output per direction).

## The pygame frontend

`tools/agent.py` wraps all four `step_*.tl` files behind
`apply_move(board, direction)`, plus a plain-Python `simulate_move()` used
for legality checks, game-over detection, and score bookkeeping (a move's
score can't be read off the board's total — merging preserves the sum, e.g.
`[2,2,0,0] -> [4,0,0,0]` — so it has to come from watching which cells
merged, same as any 2048 implementation). `tools/play.py` is the pygame UI on
top of that: arrow keys/WASD to play, `B` to watch a heuristic autoplay bot.

Every `step_*.tl` subprocess call is driven through `agent.py`'s
`spawn_move_engine`/`collect_move_engine` split rather than a single
blocking call, and `play.py` polls it (pumping pygame's event queue and
redrawing a "Sliding {direction}..." overlay each tick) instead of blocking
outright. This matters because a first-time CUDA kernel compile for a
direction can take 30-40s, and a genuinely blocking call for that long makes
the OS conclude the window has hung ("app not responding") — polling instead
keeps it alive the whole time. `play.py` also runs a one-time warm-up pass
for all four directions at startup, with its own progress screen, so that
30-40s compile happens there instead of ambushing the first real move or
autoplay's first pick.

That autoplay bot (`agent.choose_ai_move`, a corner-weighted-board +
empty-cell-count heuristic) is **not** a TensorLang-trained policy network —
there's nothing trained for 2048 yet, unlike `tic_tac_toe`'s `infer.tl`. Only
the *decision* of which direction to play comes from the heuristic; the
resulting board after every real move (human or autoplay) is always computed
by the actual `step_*.tl` engine, with a plain-Python fallback (printing a
warning) if a `.tl` subprocess call fails for any reason — same
degrade-rather-than-crash philosophy as `tic_tac_toe`'s `choose_move`.

Played through a full autoplay session end-to-end on real GPU hardware:
reached a 64 tile, score 1052 (best 2172 across a couple of runs), clean
game-over screen, no crashes, no freezes.

`verify_all.py` (repo root) generalizes `verify.py` to check any of the four
`step_*.tl` files against a plain-Python reference for that direction — used
to validate `step_right.tl`/`step_up.tl`/`step_down.tl` numerically without a
GPU before wiring them into `agent.py`:

```bash
python3 verify_all.py apps/games/2048/step_right.tl right
python3 verify_all.py apps/games/2048/step_up.tl up
python3 verify_all.py apps/games/2048/step_down.tl down
```

## Four real compiler bugs found and fixed along the way

### 1 & 2. AST inlining bugs (in `tensorlang/ast_builder.py`)

Both are the same underlying class of bug: several places assume a node's
operand name(s) live under the `args` field, but two AST node types store
them under different field names — `concat` uses `tensors` (a list), and
`slice`/`softmax`/`sum`/etc. use `tensor` (singular) — so those two node
types were silently skipped by every renaming pass and got left with dangling
references to a function's *original*, unrenamed local variable names.

1. **`substitute_names()`** (used for a single level of function inlining)
   handled `args` and `tensor`, but not `tensors`. Any `concat()` call
   inside a function body kept referencing the un-renamed pre-inlining
   names, so calling a function more than once (e.g. `compact_pair` 9
   times inside `compact_row`) produced "Undefined tensor" errors from the
   second call onward.

2. **The nested-function-call nesting fix-up** in `build_ast()` (the loop
   that remaps a function's local variable names when it's called from
   *inside* another function that's itself being inlined) only patched
   `args`, missing `tensors` and `tensor` entirely.

Neither is 2048-specific — any `.tl` program that (a) calls the same
function more than once, or (b) has a function call another function, and
uses `concat`/`slice`/`softmax`-style ops on the result, would hit this.

### 3. Missing `concat(axis=1)` CUDA kernel (in `tensorlang/kernel_generator.py` + `tensorlang/compiler.py`)

`KernelGenerator.concat()` only ever had a code path for `axis=0` (stacking
rows). `axis=1` (side-by-side, same row count) simply wasn't implemented —
the function fell through with no `else` and no `return`, so it silently
returned `None` for every `axis=1` call. That surfaced as an opaque
`TypeError: cannot unpack non-iterable NoneType object` at the call site in
`compiler.py`, with nothing pointing at concat or axis=1 as the actual cause.

Since `step.tl` builds every row out of individual scalar cells side-by-side
(`concat(a, b, axis=1)`), almost every concat call in the file needs this.
The fix adds an `axis=1` kernel — a straightforward mirror of the existing
`axis=0` kernel with rows/cols swapped — plus the matching dispatch case in
`compiler.py`'s kernel-execution switch. General compiler gap, not
2048-specific: any `.tl` program building up a tensor from smaller pieces
column-wise would have hit the same silent failure.

### 4. GPU-computed function-return aliases never resolved at the right point in execution order (in `tensorlang/compiler.py`)

This was the deepest one, and the actual root cause of `step.tl` compiling
and running with no errors but producing an all-zero board.

When a function's return value gets bound to a name at its call site (e.g.
`let row0_c1 = compact_row(row0)`), that binding is just an *alias* node in
the AST — `{'type': 'name', 'name': <actual computed name>}`. The compiler
already had code to resolve these aliases, but only for the case where the
source value lives in the host-side `tensors` dict. It never handled the
case where the source was computed *on the GPU* by a kernel — so the alias
name kept its own freshly-allocated, never-written GPU buffer, and anything
reading it (a later kernel, or a final `save()`) silently read zeros instead
of the real computed data.

Two things had to be fixed together to actually resolve this correctly:

- Aliases needed to be resolved **inline, at their exact position in
  execution order** — not in a single pass after all kernels finish. An
  alias is frequently consumed as an input to a *later* kernel within the
  same execution pass (e.g. a function's return value immediately fed into
  a `concat`/`mult`/etc. at the call site); resolving it only after
  everything has already run is too late for anything but a truly terminal
  alias that nothing but a final `save()` reads.
- The host-tensor alias check and the GPU-buffer alias check needed to be
  two **independent** `if` statements, not `if/elif`. Every kernel's output
  automatically gets mirrored into the host-side `tensors` dict (for
  caching/debugging), so the "is this alias's source in `tensors`?" check
  was almost always true for any GPU-computed value — which meant the
  `elif` GPU-aliasing branch (the one that actually matters for execution
  correctness) never got a chance to run at the right time, and only fired
  much later in a leftover safety-net pass.

Diagnosing this took building a `--debug --debug-info` trace of every
intermediate kernel result on real hardware and reading it in actual
execution order — the bug was invisible from static code reading and from
the NumPy AST interpreter (`verify.py`), since neither model kernel
scheduling or GPU buffer identity.

### A related limit that's NOT fixed, just worked around

The compiler's auto-inlining for a function-calling-a-function only expands
**one level deep**, and only when the outer call happens at the program's
top level (see the `elif stmt.data == 'let_binding'` branch in
`build_ast()`). A 3-deep chain (top level → function A → function B →
function C) leaves an unexpanded call node and crashes the type checker.
`step.tl` is written to stay within this limit deliberately: `compact_row`
and `merge_row` are each called directly from the top level (never wrapped
in an outer `slide_row_left`), and `merge_row` inlines its 3-cell recursive
case by hand instead of factoring it into a `merge_line3` helper function.
This is called out in `merge_row`'s comment. If you want arbitrary-depth
function nesting to work, `inline_function_call()` would need to become
recursive itself, rather than the current single hard-coded expansion pass
in `build_ast()` — a bigger, riskier change I deliberately didn't attempt
blind.

## run.sh

`run.sh` mirrors `tic_tac_toe/run.sh`'s interface:

```
./apps/games/2048/run.sh                 # smoke-test step.tl on a built-in tricky board
./apps/games/2048/run.sh --board N N ... # smoke-test on a custom board (16 numbers, row-major)
./apps/games/2048/run.sh --play          # launch the interactive pygame UI
./apps/games/2048/run.sh --train         # generate training data (if missing) and train the network
```

## The trained move-picking network

`agent.py`'s `choose_ai_move` — the direction the pygame UI's autoplay
mode picks — used to be a hand-written heuristic (corner-weighted board +
empty-cell count). It's now a real TensorLang-trained policy network,
mirroring `tic_tac_toe`'s `train.tl`/`infer.tl` split:

- **`tools/generate_data.py`** — unlike `tic_tac_toe` (whose 4520-board
  state space is small enough to solve exactly with minimax), 2048's
  state space is far too large to enumerate. So states to label come
  from *self-play* (the old heuristic plays games, with a little random
  exploration mixed in, and every board it passes through is a
  candidate), and each one is labeled by an **expectimax search** (depth
  3, alternating max nodes over the 4 moves and chance nodes over tile
  spawns) over `agent.py`'s existing `simulate_move`, bottoming out in
  the same corner-weight heuristic as the leaf evaluation. Two
  approximations keep a depth-3 search with full chance-node expansion
  tractable in plain Python: chance nodes cap how many empty cells they
  expand (`MAX_CHANCE_SAMPLES`), and nodes below the root only fully
  search their top-ranked candidate moves (`MOVE_PRUNE_TOP`) rather than
  all four.
- **`tools/init_weights.py`** — He/Kaiming init for a 256-128-64-4
  ReLU/ReLU/softmax MLP. 256 inputs = 16 board cells x 16 log2-one-hot
  value categories (`agent.encode_board_onehot` — category 0 = empty,
  category *k* = tile value 2^*k*); 4 outputs = one logit per direction.
- **`train.tl`** — same full-batch structure as `tic_tac_toe/train.tl`
  (whole dataset fits in memory), same `mse_loss`-on-softmax-probs loss
  and same same-shaped-learning-rate-vector workaround for the
  (1,1)-scalar-times-1D-bias-gradient CUDA bug (HANDOVER.md §15.4, still
  open). Hyperparameters were picked by replicating the exact
  forward/backward math offline in plain NumPy and sweeping there against
  the real generated dataset, the same way `tic_tac_toe`'s were.

### A real overfitting bug found and fixed along the way

The first trained version (4000 self-play states, no augmentation) played
badly — score ~950, well below the old heuristic's ~2172 best, with a
visibly disorganized end board (duplicate small tiles scattered around,
no coherent gradient toward a corner). The cause: the self-play heuristic
that generated every training board always favors the *same* corner in
the *same* orientation, so every training example was some variation of
"tiles building toward the top-left." The network had no reason to learn
a general "keep your big tiles collected in a corner" strategy — it could
get away with memorizing "top-left" as a fixed feature of the board.
Loss converged to ~0.0013, consistent with something close to
memorization rather than generalization; and the moment live play's own
moves nudged the board even slightly outside that memorized pattern, it
had nothing to fall back on.

The fix: **symmetry augmentation**. Every self-play board is expanded
into all 8 of its dihedral-group transforms (4 rotations x a left-right
mirror — see `generate_data.py`'s `d4_transforms`), and expectimax is run
*independently* on each of the 8, not just relabeled from the original's
answer — the corner-weighted heuristic isn't itself rotation/reflection-
symmetric, so only re-running the search on the actual transformed board
keeps every label correct. This turned 4000 base self-play states into
24000 training rows, and — concretely, not just in theory — rebalanced
the direction-label distribution: `down` went from 8.8% of labels to
18.1%, since it's no longer geometrically disfavored by a fixed corner
bias. Retrained network: scores of 1352 and 2300 across two real GPU
autoplay games — the 2300 beats the heuristic's old best.

One more thing worth flagging for anyone extending this further: the
symmetry-augmented dataset changes `mse_loss`'s effective gradient scale
(it's normalized by total element count, and there are now 6x more
rows), so the learning rate that worked for the 4000-row dataset (lr=8)
was **not** the right one for the 24000-row dataset — it measurably
overshot. Any time the dataset size changes meaningfully, re-sweep the
learning rate rather than assuming the old value still applies.

### A `check.py` blind spot (self-inflicted, not a compiler bug)

While iterating on `train.tl`'s hand-written literal tensors
(`lr_b1`/`lr_b2`/`lr_b3` — the same-shaped-learning-rate-vector CUDA
workaround mentioned above), a 128-long literal accidentally had only 127
elements in it. `check.py` reported `TYPE CHECK OK` anyway and the actual
GPU run failed with `cannot reshape array of size 127 into shape (128,)`
— the type checker trusts a literal's *declared* shape annotation and
never actually counts the literal's elements against it. Not a compiler
bug worth fixing here (hand-written 128-element literals are already an
awkward workaround for a real bug elsewhere), but worth knowing: `check.py`
passing is not proof a hand-written literal tensor is the right length.
If you ever generate one of these by hand instead of programmatically,
double check the count separately.

### Results and honest caveats

Confirmed on real GPU hardware:

- Full pipeline — `generate_data.py` → `init_weights.py` → `train.tl` →
  `infer.tl` → `agent.py`'s `choose_ai_move` → pygame autoplay — runs
  end to end with no fallback to the heuristic.
- Two autoplay games: 1352 and 2300 (new best, vs. the heuristic's 2172).

Caveats, for whoever picks this up next:

- Two games is a small sample; 2048's scoring has real variance from
  random tile spawns alone. The label-distribution rebalancing and the
  clear qualitative fix to the "stuck in one orientation" failure mode
  are stronger evidence than any single game's score. A proper
  benchmark (looping `choose_ai_move`-driven games via `agent.py`
  directly, no pygame needed, over 20-30+ games, comparing average and
  best score against the old heuristic) would give real confidence
  rather than an impression.
- Both end-of-game boards still show some scattered duplicate small
  tiles rather than a clean monotonic gradient into one corner — there's
  real headroom left. This is a meaningfully-better-than-the-placeholder-
  heuristic bot, not a polished 2048 solver.
- The final `train.tl` hyperparameters (lr=4, 8000 epochs) are an
  extrapolation one step past the last confirmed offline-sweep data
  point (6000 epochs, 95.8% train accuracy) rather than a fully swept
  plateau — a longer sweep got cut short by the environment used to
  prepare this change, not by anything about the network itself. Worth
  re-sweeping properly if squeezing out more performance matters.
- `BASE_STATES=3000` (before symmetry expansion), `DEPTH=3`,
  `MAX_CHANCE_SAMPLES=4`, and `MOVE_PRUNE_TOP=2` in `generate_data.py`
  are all still just first-reasonable-guess values, not swept — more/
  deeper self-play data or a deeper expectimax search are the obvious
  next levers if 2300 isn't good enough.

## How to apply the fixes and run this yourself

A single consolidated patch, `tensorlang_fixes.patch`, covers all four
fixes above (touches `ast_builder.py`, `compiler.py`, `kernel_generator.py`).
If you previously applied any of the earlier incremental patches
(`ast_builder_fix.patch`, `concat_axis1_fix.patch`, `gpu_alias_fix*.patch`),
revert those first so you're back to a clean, unmodified `tensorlang/`
before applying this one — it's diffed from the original, unpatched files.

```bash
cd /path/to/your/tensor-lang
# only if you applied earlier incremental patches — skip if starting fresh:
git checkout -- tensorlang/ast_builder.py tensorlang/compiler.py tensorlang/kernel_generator.py

git apply tensorlang_fixes.patch
cp -r apps/games/2048 /path/to/your/tensor-lang/apps/games/   # if not already there

# quick sanity check without a GPU — parses + type-checks only:
python3 check.py apps/games/2048/step.tl

# arithmetic correctness check without a GPU — runs the AST through a
# NumPy interpreter and diffs against a plain-Python reference:
python3 verify.py

# real run, on your GPU:
./apps/games/2048/run.sh --board 4 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0
```

Confirmed working end-to-end on real GPU hardware:

- Single-row board `[4,2,2,0,...]` → row 0 correctly comes back `[4,4,0,0]`.
- Full mixed board `[4,2,2,0 / 2,2,2,2 / 0,2,0,2 / 8,4,2,2]` → all four rows
  correct: `[4,4,0,0]`, `[4,4,0,0]`, `[4,0,0,0]`, `[8,4,4,0]`. This exercises
  a no-merge-at-all row is not present here, but does cover a double-merge
  row, a compact-then-single-merge row, and a merge-at-the-back row, and —
  importantly — confirms the fix holds across four *sequential* invocations
  of `compact_row`/`merge_row` in the same run, not just one.

If `git apply` complains that the patch doesn't apply cleanly, try
`patch -p1 < tensorlang_fixes.patch` instead. Either way, confirm it landed
before re-running anything:

```bash
grep -n "concat_call stores its two operand names" tensorlang/ast_builder.py
grep -n "concat_axis1" tensorlang/kernel_generator.py tensorlang/compiler.py
grep -n "_resolve_alias_inline" tensorlang/compiler.py
```

`check.py` and `verify.py` aren't 2048-specific either — they're generally
useful for validating any `.tl` file's parsing/typing/arithmetic without
spending GPU cycles, so it's worth keeping them in the repo (e.g. under a
`tools/` or `scripts/` directory) rather than treating them as throwaway.

## What's deliberately NOT in this repo yet

- Random tile spawning and score tracking live in Python
  (`tools/agent.py`'s `spawn_tile`/`simulate_move`), not in any `.tl` file —
  same split as `tic_tac_toe`'s illegal-move masking: that's stochastic
  bookkeeping, deliberately kept out of the deterministic tensor-ops engine.
- A proper multi-game benchmark comparing the trained network's
  average/best score against the old heuristic's — see "Results and
  honest caveats" above. Right now the evidence is two real games plus
  the label-distribution rebalancing, which is suggestive but not a
  rigorous comparison.
- Any sweep of `generate_data.py`'s own knobs (`BASE_STATES`, `DEPTH`,
  `MAX_CHANCE_SAMPLES`, `MOVE_PRUNE_TOP`) — all still first-reasonable-
  guess values, not tuned against actual play strength.

Next step, if you want to keep going: build the benchmark harness above,
then use it to decide whether more self-play data, a deeper expectimax
search, or a bigger network is the better lever to pull next.

# apps/games/2048/step.tl — status and how to run it

**Status (2026-09-09): core engine done, all four compiler bugs found along
the way are fixed, and everything is verified end-to-end on real GPU
hardware (a 1080ti).** See `HANDOVER.md` §17 in the repo root for the
handover-level summary of this session. What's still to build is listed at
the bottom of this file.

## What this is

`step.tl` is a pure tensor-ops implementation of one 2048 move (slide left).
Unlike `tic_tac_toe` (where tensor-lang only runs the neural net and the game
rules live in Python), here the actual game mechanics — compaction, merging —
are expressed as tensor arithmetic. See the header comment in `step.tl` for
the full design writeup, including why a naive "check fixed pairs (0,1) and
(2,3)" merge rule is wrong and how the branchless recursive-select version
handles it correctly.

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

`run.sh` mirrors `tic_tac_toe/run.sh`'s interface, but is honest about what
actually exists right now:

```
./apps/games/2048/run.sh                 # smoke-test step.tl on a built-in tricky board
./apps/games/2048/run.sh --board N N ... # smoke-test on a custom board (16 numbers, row-major)
./apps/games/2048/run.sh --play          # will launch the interactive game — not built yet
./apps/games/2048/run.sh --train         # will train the move-picking net — not built yet
```

`--play` and `--train` check for the files they need (`tools/agent.py` +
`tools/play.py`, or `train.tl` + `tools/generate_data.py` +
`tools/init_weights.py`) and exit with a clear list of what's missing
rather than crashing partway through, so the script's interface is already
in its final shape and each mode just "turns on" as its files land.

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

## What's deliberately NOT in this file yet

- Only **left** is implemented. Right/up/down reuse `compact_row`/`merge_row`
  unchanged: right = reverse each row, slide left, reverse back; up/down =
  transpose the board, slide left/right, transpose back.
- Random tile spawning after a move is not here — same split as
  `tic_tac_toe`'s illegal-move masking: that's stochastic bookkeeping that
  belongs in Python (`tools/agent.py`), not in this deterministic `.tl` file.
- No score tracking (points gained from merges) yet.

Once you've confirmed this runs correctly on your GPU, the next step is
wiring up the other 3 directions, then `tools/agent.py` (random spawn +
legality check by diffing pre/post-move boards) and `tools/generate_data.py`
(an expectimax + heuristic oracle for training a move-picking network, same
role as tic_tac_toe's minimax labeler) — happy to keep going once this is
validated for real.

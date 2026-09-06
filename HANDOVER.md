# TensorLang Handover Document

**Last updated:** 2026-09-06
**Project:** [davro/tensor-lang](https://github.com/davro/tensor-lang)
**Status:** Core language + test suite healthy (109/109). App system restored and demonstrated with two working apps (`hello_mlp`, `decision_boundary`). New shared toolkit (`apps/tlkit/`) for building visual/interactive apps. §12 fixes from 2026-09-03 all still holding. Of the five compiler/runner limitations discovered while building `decision_boundary` (§13.3), four are now fixed upstream and hardware-verified (§15); the fifth uncovered a second, still-open issue in CUDA kernel dispatch for scalar broadcasting (§15.4).

---

## 1. What TensorLang Is

TensorLang is a small native ML language that:

- Parses `.tl` source
- Type-checks tensor shapes and broadcasting
- Emits optimised CUDA kernels (forward + backward)
- Compiles with `nvcc` and runs via PyCUDA
- Supports reverse-mode automatic differentiation and multi-step training loops

Goal: eliminate Python interpreter overhead for tensor / neural-network workloads by compiling directly to GPU kernels while keeping an ML-friendly syntax.

**Current version shown in compiler output:** TensorLang 0.3.0

---

## 2. Quick Start (Fresh Machine)

```bash
git clone https://github.com/davro/tensor-lang.git
cd tensor-lang

# Install (run with bash, not source — see note below)
bash build.sh --install

# Activate the venv
source python-env/bin/activate

# Full test suite
source build.sh --test
# or
python3 tensorlang.py --test
```

**Important notes**

- Always prefer `bash build.sh --install`. Using `source build.sh --install` hits an `exit 0` that kills the current shell.
- `pycuda` requires a working NVIDIA CUDA toolkit.
- After install, the venv lives in `python-env/`.

---

## 3. Repository Layout (Key Parts)

```
tensor-lang/
├── tensorlang.py              # CLI entry point
├── tensorlang/                # Core package
│   ├── app_runner.py          # Application runner (restored 2026-08-29)
│   ├── ast_builder.py
│   ├── autograd.py
│   ├── compiler.py
│   ├── kernel_generator.py
│   ├── tensor_lang.py         # Argument parsing etc.
│   ├── tensor_verifier.py
│   ├── test_runner.py
│   └── type_checker.py
├── tensorlang.lark            # Grammar
├── build.sh                   # Install / test / lint helper
├── tests/                     # 109 core language tests
├── apps/                      # User-facing applications
│   ├── tlkit/                 # Shared toolkit for visual/interactive apps (new, 2026-09-04)
│   │   ├── chunked_runner.py  # drives repeated `tensorlang.py --app` invocations
│   │   ├── sequence_player.py # pygame play/pause/scrub/speed harness
│   │   ├── colormap.py        # numpy colormaps + 2D grid generation
│   │   └── README.md
│   └── examples/
│       ├── hello_mlp/         # Minimal working app
│       │   ├── app.toml
│       │   └── main.tl
│       └── decision_boundary/ # 2nd app: XOR MLP + pygame boundary animation (new, 2026-09-04)
│           ├── app.toml
│           ├── main.tl
│           ├── run.sh         # install pygame, init/train/view in one command
│           ├── data/          # static grid + metadata (checked in)
│           ├── snapshots/     # generated animation frames (gitignored)
│           └── tools/         # init_state.py, train_and_snapshot.py, viewer.py
├── examples/                  # Older / misc examples
├── cache/                     # Generated CUDA, .npy, etc. (gitignored)
└── requirements.txt
```

---

## 4. Current Health (2026-09-03)

| Component                         | Status                                                             |
|------------------------------------|---------------------------------------------------------------------|
| Core test suite (default, parallel) | **106/106 passed** (~94 s, re-confirmed after all §12 fixes)      |
| Core test suite (`--no-parallel`)  | **Fixed & verified on hardware** (2026-09-03) — see §12              |
| Default compiler output (no flags) | **Fixed & verified on hardware** (2026-09-03) — see §12              |
| AppRunner                          | Restored and working for normal app execution                       |
| App test mode (`--app X --test`)   | **Fixed & verified on hardware** (2026-09-03) — see §12               |
| Example app (1st)                  | `apps/examples/hello_mlp` runs successfully                         |
| Example app (2nd)                  | `apps/examples/decision_boundary` — verified on hardware (2026-09-04); loss 0.244 → 0.0016 over 1000 epochs, see §13.2 |
| Shared app toolkit                 | `apps/tlkit/` — extracted from `decision_boundary`, unit-smoke-tested, see §13.1 |
| Autograd + training loops          | Working (verified by hello_mlp + decision_boundary + tests)         |
| Missing file history                | `app_runner.py` was absent from main; restored from previous work   |

**Known temporary fix applied earlier:**
The import of `AppRunner` had to be commented out until the file was restored. It is now present again.

**Recent fixes (2026-09-03):** Three issues were found during a code review of `tensorlang.py`, `app_runner.py`, `test_runner.py`, and `compiler.py`, and patched in `test_runner.py`, `compiler.py`, and `type_checker.py`. Full detail, root cause, and verification status are in §12.

---

## 5. How to Run Things

### Tests
```bash
source build.sh --test
source build.sh --test --filter for_loop
source build.sh --test --filter autograd
python3 tensorlang.py --test --filter mlp
```

### Single file
```bash
python3 tensorlang.py --cache-layers tests/training_pipeline.tl
python3 tensorlang.py --debug --cache-layers path/to/file.tl
```

### Applications
```bash
python3 tensorlang.py --list-apps          # if CLI flag is wired
python3 tensorlang.py --app examples/hello_mlp
```

---

## 6. Application System (`apps/`)

Apps are discovered by the presence of an `app.toml`.

### Required layout
```
apps/
└── [<category>/]
    └── <app-name>/
        ├── app.toml          # required
        ├── <entry>.tl        # main / train entry point
        └── tests/            # optional
            └── *.tl
```

### Minimal `app.toml`
```toml
[app]
name = "hello_mlp"
description = "Minimal 2-layer MLP training example (linear regression style)"

[requirements]
gpus = 1
memory_gb = 4

[entry_points]
main = "main.tl"
# train = "train.tl"          # fallback if main missing
# benchmark = "bench.tl"      # optional
```

### Example that currently works
`apps/examples/hello_mlp` trains a single weight to fit `y = 2x` over 5 epochs and produces the expected convergence:

- Final loss ≈ 0.01648
- Final weight ≈ 2.046875

This is the canonical "hello world" for the app system.

### Second app: a visual, interactive-feeling example

`apps/examples/decision_boundary` (added 2026-09-04) trains a 2-8-1 tanh/sigmoid MLP on XOR and plays back the decision boundary morphing across training as a pygame animation. It's a genuinely different shape of app from `hello_mlp` — a single training run isn't enough here, the point is to *see* training happen — and it surfaced real constraints in the compiler that a single-file, single-run app never would. See §13 for the full writeup, including the five compiler/runner gotchas found along the way and the general-purpose `apps/tlkit/` toolkit extracted from it.

Run it:
```bash
./apps/examples/decision_boundary/run.sh          # installs pygame if missing, trains, plays back
./apps/examples/decision_boundary/run.sh --reset  # wipe weights and start fresh
```

---

## 7. Language Features (Implemented)

- Explicit tensor types: `Tensor[f32, (batch, features)]`
- 40+ operations (matmul, element-wise, activations, norms, losses, reductions, slicing, etc.)
- Reverse-mode autodiff (`with grad` + `backward()`)
- Training loops with weight rebinding (`w = w_updated`) — pointer swap, no copy
- Inline binary expressions (flattened to temporaries)
- User-defined functions
- Control flow (`if`/`elif`/`else`)
- File I/O (`load` / `save`)
- Caching of kernels and intermediate tensors

**Important constraint:**
All kernels from a compilation unit (including loop bodies) go into one `.cu` file. Tensor names used inside a `for` loop body cannot be reused at top level (use `y_pred_final` etc.).

---

## 8. Test Inventory (current `tests/`)

The suite covers:

- Basics, arithmetic, broadcasting
- Activations (relu, sigmoid, tanh, softmax, …)
- Reductions, slicing, reshape, transpose, concat
- Linear layers, MLP, neural nets
- Normalisations (layer/batch/instance)
- Losses (mse, cross-entropy)
- Autograd (basic, chain, broadcast, relu, sigmoid, tanh, weight update)
- Training loops (`for_loop_*`, `training_pipeline`, `linear_regression`, …)
- Functions, pipelines, portfolio-style examples
- Load/save

Full list is available via `ls tests/`. All 109 currently pass (106 + 3 added in §15.2).

---

## 9. Architecture (Compilation Pipeline)

1. **Lex/Parse** – Lark grammar (`tensorlang.lark`)
2. **AST construction** – flatten inline expressions, inline user functions
3. **Type checking** – shapes, broadcasting, gradient tensor pre-registration
4. **CUDA generation** – forward + backward kernels
5. **Compile** – `nvcc` → shared library
6. **Execute** – PyCUDA launch, memory management, loop iteration with pointer-swap rebinding

Key modules: `compiler.py`, `kernel_generator.py`, `type_checker.py`, `autograd.py`, `ast_builder.py`.

---

## 10. Known Pain Points & Future Work

From README + recent experience:

**Short term**
- Scoped kernel names (remove post-loop renaming constraint)
- ~~Inline unary ops (`relu(linear(...))`)~~ — **done, see §15.1**
- Better shape-mismatch error messages
- ~~Cleaner / quieter compiler output for app runs~~ — **done, see §12.3**
- **New (2026-09-04, see §13 for full detail):**
  - ~~`mult`/`add`/`minus`/`div`'s 2D-vs-1D broadcast branch only special-cases a size-1 vector correctly; a same-shape-but-larger 1D operand (e.g. an 8-wide bias gradient against a `(1,1)` scalar) hits a false "shape mismatch"~~ — **type checker fixed 2026-09-06, see §15.1; CUDA kernel dispatch for this case still needs a fix, see §15.4**
  - ~~`reshape()` does a standalone env lookup with no `_grad`-suffix inference, unlike `add`/`minus`/`mult`/`div` — it can never take a gradient tensor directly, independent of shape~~ — **done, see §15.1** (fixed for every single-tensor-field op, not just `reshape`)
  - ~~`app_runner.py` calls `compiler.compile_and_execute()` without checking its return value, so a failed compile/execute can still exit `0` — scripts driving `tensorlang.py` as a subprocess cannot trust the exit code alone~~ — **done, see §15.1**

**Medium term**
- Built-in optimisers (SGD, Adam) as language primitives
- Real implementation of AppRunner requirement validation
- Hot-reload in dev mode
- Kernel fusion

**Product / motivation**
- More practical example applications (vision, simple game agent, signal processing, finance) so the language does not feel purely abstract.

---

## 11. Recent Recovery Notes (Aug 2026)

- Fresh clone + `bash build.sh --install` succeeded.
- `ModuleNotFoundError: tensorlang.app_runner` occurred because the file was missing from main.
- File was restored from prior local work; import re-enabled.
- Full test suite went green (106/106).
- First app (`examples/hello_mlp`) runs end-to-end and matches documented training numbers.

---

## 12. Known Issues & Recent Fixes (2026-09-03)

Two bugs were found in `test_runner.py` during a code review (triggered by
tracing how `AppRunner._run_app_tests` wires into `TestRunner`, and by
inspecting the sequential branch of `run_test_suite`). Both were pre-existing
bugs, not introduced by anything in this session's changes — the second one
in particular means **app-level tests have likely never worked** since
`AppRunner` was written.

### 12.1 `--no-parallel` crashed with a `TypeError` — FIXED, verified on hardware

**Symptom (confirmed on real hardware):**
```
python3 tensorlang.py --test --no-parallel
Running 106 tests sequentially...
Unexpected error: TestRunner.run_single_test() got multiple values for argument 'suite_start_time'
```

**Root cause:** `run_test_suite`'s sequential branch called:
```python
self.run_single_test(test_file, self.verify_tensors, suite_start_time=None, debug_mode=self.debug_mode)
```
but `run_single_test`'s signature is `(self, test_file, suite_start_time=None)`.
`self.verify_tensors` was landing in the `suite_start_time` positional slot,
clashing with the explicit `suite_start_time=None` keyword, and `debug_mode`
isn't a parameter at all. `verify_tensors` and `debug_mode` are already read
from `self` inside `run_single_test`, so the extra arguments were never
needed.

**Fix:**
```python
test_result = self.run_single_test(test_file, suite_start_time=None)
```

**Verification (hardware, 2026-09-03):**
```
python3 tensorlang.py --test --no-parallel --filter sum
Running 3 tests sequentially...
================================================================================
State TestCase      Time
--------------------------------------------------------------------------------
PASS   sum_axis.tl    (3.53s)
PASS   sum_axis0.tl   (3.57s)
PASS   sum_full.tl    (3.42s)
================================================================================
Summary: 3/3 tests passed in 10.52s
✅ All tests passed successfully!
```
Filtered run used to speed up validation; a full `--no-parallel` run (106
tests, ~100s+ since sequential is slower than the parallel default) is still
worth doing once, but the crash itself is confirmed gone.

### 12.2 `--app <name> --test` crashed with a `TypeError` — FIXED, verified on hardware

**Root cause:** `AppRunner._run_app_tests` constructs:
```python
TestRunner(parallel=False, verify_tensors=..., debug_mode=..., tests_dir=str(tests_dir))
```
but `TestRunner.__init__` didn't accept a `tests_dir` parameter at all —
guaranteed `TypeError` before a single app test could run. Separately, even
if the constructor accepted it, `discover_tests()` and `run_single_test()`
both hardcoded `Path("tests")` / `"tests/{test_file}"`, so app-scoped tests
under `apps/<name>/tests/*.tl` could never be found or executed through this
path — `AppRunner` and `TestRunner` had drifted out of sync.

**Fix (in `test_runner.py`):**
- `TestRunner.__init__` now takes `tests_dir=None`, defaulting to `"tests"`
  (preserves existing core-suite behavior).
- `discover_tests()` now globs `Path(self.tests_dir)` instead of the
  hardcoded `Path("tests")`.
- `run_single_test()` now builds the `.tl` path, the `tensorlang.py`
  subprocess command, and the cache directory from `self.tests_dir` instead
  of a hardcoded `"tests/"` string. Cache now lands at
  `cache/<tests_dir>/<test_file>/` — so app tests get their own cache
  location instead of colliding with the core suite's `cache/tests/`.
- The failure-log path printed in `print_results_table`/summary output also
  now respects `self.tests_dir`.

`app_runner.py` itself needed **no changes** — it was already calling the
intended API; `TestRunner` just hadn't been built to match yet.

**Also added:** neither `apps/examples/hello_mlp` nor
`apps/examples/linear_regression` had a `tests/` subdirectory, so there was
nothing for the fix to discover. Added one test each, with `@EXPECTED`
values taken directly from real hardware output (not rounded
approximations) and independently cross-checked against a from-scratch
numpy re-implementation of the same full-batch gradient descent at float32
precision (bit-exact match in both cases):
- `apps/examples/hello_mlp/tests/loss_final.tl` — expects `loss_final = 0.0164794921875`
- `apps/examples/linear_regression/tests/loss_final.tl` — expects `loss_final = 0.022800864651799202`

**Verification (hardware, 2026-09-03):**
```
python3 tensorlang.py --app examples/hello_mlp --test
Running tests from: apps/examples/hello_mlp/tests
Running 1 tests sequentially...
================================================================================
State TestCase       Time
--------------------------------------------------------------------------------
PASS   loss_final.tl   (3.58s)
================================================================================
Summary: 1/1 tests passed in 3.58s
✅ All tests passed successfully!

python3 tensorlang.py --app examples/linear_regression --test
Running tests from: apps/examples/linear_regression/tests
Running 1 tests sequentially...
================================================================================
State TestCase       Time
--------------------------------------------------------------------------------
PASS   loss_final.tl   (4.03s)
================================================================================
Summary: 1/1 tests passed in 4.03s
✅ All tests passed successfully!
```
Both apps' training loops are now independently confirmed mathematically
correct (autograd, `mse_loss`, `backward()`, weight-rebind pointer-swap in
`for` loops), not just "didn't crash" — and the app-test wiring fix is
confirmed working end-to-end on real hardware.

### 12.3 Default compiler output was extremely noisy — FIXED, verified on hardware

**Symptom:** a plain `python3 tensorlang.py file.tl` (no flags) printed 40+
lines before showing anything useful — a full ASCII banner, a file-details
block, "Loaded Lark Grammer file"/"Loaded TensorLang file", the entire raw
`.tl` source dumped back to stdout, a pipeline banner, a `[COMPILER] Result`
block for *every* intermediate tensor (not just the final one), and a
content-free `[COMPILER] KERNEL CUDA!` line. This is the item listed under
§10 "Cleaner / quieter compiler output for app runs" — now resolved.

**Root cause:** none of this output was gated behind `self.debug_mode`; it
all fired unconditionally on every run, regardless of flags.

**Fix:** gated everything above behind `self.debug_mode` in `compiler.py`
(`--debug` still shows exactly the original verbose output — confirmed
byte-for-byte identical). Also fixed `type_checker.py`'s
`[TYPE CHECKER] Arg Names`/`Args` trace, which printed unconditionally for
every `let` binding — now gated behind `DEBUG_MODE` to match the identical
pattern already used one line below it.

Added an explicit final-result print using `output_tensor` — the AST's
designated result name, returned by `build_ast` but previously never
actually looked up or printed. Before this fix, the "final answer" a user
saw was really just whichever intermediate `Result` print happened to run
last (coincidentally correct for single-output programs, not by design).

**New default output:**
```
[COMPILER] TensorLang 0.3.0 — compiling <file>
<output_tensor> = <value>
Done in <elapsed>s
```

**Verification (hardware, 2026-09-03):**
```
python3 tensorlang.py --test
================================================================================
Summary: 106/106 tests passed in 93.86s
✅ All tests passed successfully!
================================================================================

python3 tensorlang.py tests/activation_pipeline.tl
[COMPILER] TensorLang 0.3.0 — compiling activation_pipeline.tl
output = [0.31830028 0.5        0.68169975 0.7239275 ]
Done in 2.73s
```
`--debug` confirmed to reproduce the full original verbose trace unchanged.
Full core suite re-run confirms this change (and the §12.1/§12.2 fixes) did
not affect pass/fail behavior — `test_runner.py` verifies against `.npy`
cache files, not stdout, so none of this touches correctness.

---

## 13. Session 2026-09-04: `apps/tlkit/`, second app (`decision_boundary`), and five newly discovered compiler/runner limitations

Builds directly on §14 action #6 ("pick one concrete practical application"). Delivered as a Claude-assisted session; every fix below was verified either by parsing the real `tensorlang.lark` grammar directly, by hand-simulating the exact `type_checker.py` branch logic against real shapes, or on real GPU hardware (the loss curve in §13.2) — not by inspection alone.

### 13.1 `apps/tlkit/` — shared toolkit for visual/interactive apps

`hello_mlp`-style apps are one-shot: compile, train, print a number. `decision_boundary` is a different shape of app — the point is to *watch* training happen — and that need is going to recur (a physics-playback app and an interactive click-to-retrain app are both queued up next — see §14). Rather than let each new example reinvent the same plumbing, the reusable pieces were extracted into `apps/tlkit/`:

- **`chunked_runner.py`** — drives repeated `python3 tensorlang.py --app ...` invocations (see §13.2 for why chunking is necessary at all) and stacks one collected frame per chunk into a single array. Also computes `cache_dir_for(app_category_path)` so callers don't hand-write the `cache/apps/.../main.tl/...` path themselves.
- **`sequence_player.py`** — a pygame play/pause/scrub/speed-control harness generic over any `(F, ...)` frame array.
- **`colormap.py`** — diverging/sequential numpy colormaps, 2D grid generation, and pixel-coordinate mapping for overlays.

`decision_boundary`'s three tool scripts shrank from ~90 lines each to ~40–50 after the extraction. The extraction itself caught two real bugs that existed in the original, pre-extraction code and had never been exercised by a realistic test case:
- `apps/tlkit/__init__.py` eagerly importing `sequence_player` gave every tool script (including ones with nothing to do with pygame) a hard `pygame` dependency — fixed by deferring `import pygame` to inside `SequencePlayer`'s methods.
- `train_and_snapshot.py`'s `float(np.load(loss_path))` throws on the `(1,1)`-shaped array `mse_loss` actually produces (only worked by luck against an earlier, loosely-shaped `(1,)` test fixture) — fixed with `.item()`.

### 13.2 `apps/examples/decision_boundary` — XOR MLP with pygame boundary playback

Trains a 2-8-1 tanh/sigmoid MLP on XOR (`matmul`, `tanh`, `sigmoid`, `mse_loss`, `add` for bias, autograd) and animates the decision boundary morphing across training.

**Why this app is chunked into repeated process invocations instead of one long-running loop:** `save()` targets are fixed string literals in the grammar (`save_statement: "save" "(" NAME "," STRING ")"` — no `save(x, f"frame_{i}.npy")`), and rebind (`w = w_updated`) requires a fixed-shape GPU buffer every iteration, so a single `main.tl` cannot accumulate a growing snapshot history inside one `for` loop. The pattern used instead: `main.tl` `load()`s/`save()`s its own weights to a fixed cache path every run, and gets invoked repeatedly as separate processes via `tlkit.chunked_runner`, each one a "chunk" that resumes from the last.

**Verified result on hardware:** starting from a fresh init, loss dropped from 0.243585 (chunk 1) to 0.001573 (chunk 40) over 1000 total epochs (40 chunks × 25 epochs), producing a fully-separated, sharp XOR decision boundary in the pygame viewer. Getting here took several rounds of real hardware failures, documented below because each one is a genuine, reusable finding about the compiler, not an app-specific mistake.

### 13.3 Five gotchas found while building it (candidates for upstream fixes)

> **Status update (2026-09-06): four of these five are now fixed — see §15 for the fix writeup, hardware verification, and new regression tests. Left the original diagnostic text below intact since it's still the accurate record of how each was found.**

**1. Activation functions only accept a bare `NAME` argument, never a nested call. — FIXED, see §15.1**
`sigmoid_call: "sigmoid" "(" NAME ")"` and `tanh_call: "tanh" "(" NAME ")"` in the grammar — unlike `matmul`/`add`/`minus`/`mult`/`div`, which accept `inline_expr` and so support nesting. `tanh(matmul(x, w1))` fails to parse; must be split into `let h_pre = matmul(x, w1)` / `let h = tanh(h_pre)`. Confirmed by parsing both forms against the real grammar directly with `lark`.

**2. `app_runner.py` doesn't propagate `compile_and_execute`'s outcome into the process exit code. — FIXED, see §15.1**
Line ~186 calls `compiler.compile_and_execute(str(main_file))` without checking or using its return value. `compile_and_execute` has failure paths that return `False, env` without raising (e.g. a shape mismatch) — so `python3 tensorlang.py --app X` can exit `0` on a genuinely failed compile. Any script (`apps/tlkit/chunked_runner.py` included) that shells out to `tensorlang.py` and trusts the exit code alone will silently proceed past a real failure. Worked around in `chunked_runner.run_chunks` by showing the subprocess's captured stdout/stderr whenever the expected output artifact is missing, regardless of exit code.

**3. `mult`/`add`/`minus`/`div`'s 2D-vs-1D broadcast branch is too narrow. — PARTIALLY FIXED, see §15.1 and §15.4**
In `type_checker.py`, the shared branch for these four ops has a special case for `len(shape1) == 2 and len(shape2) == 1` that only succeeds if the 1D operand has size 1, or if one of `shape1`'s two dims happens to equal the 1D operand's size. A `(1,1)` scalar times an `(8,)` vector (scaling an 8-wide bias gradient by a scalar learning rate) hits neither condition and fails with `mult shape mismatch`, even though it's a completely ordinary scalar-broadcast. Confirmed by extracting and running the exact branch logic against real shapes in isolation. **Workaround used:** give each differently-shaped parameter its own same-shape learning-rate vector (`lr_b1: Tensor[f32,(8,)]`, `lr_b2: Tensor[f32,(1,)]`) so the `mult` is a trivial equal-shape operation instead of a scalar broadcast. **Update:** the type-checker half of this is fixed, but fixing it exposed a second, deeper bug in the CUDA kernel dispatch — see §15.4. The workaround above is still needed until that's resolved.

**4. `reshape()` cannot take a gradient tensor (`*_grad`) as input, regardless of shape. — FIXED, see §15.1**
`add`/`minus`/`mult`/`div` share code that, when an argument name isn't found directly in the type-checker's env, checks for a `_grad` suffix and infers the shape from the base tensor. `reshape()`'s branch does a plain `if tensor_name not in env` check with no such fallback (same is true of `transpose`, `max`/`min`/`argmax`/`argmin`, `instance_norm`, `batch_norm` — all the single-`tensor`-field ops, as opposed to the `args`-list ops). `reshape(b1_grad, (1, 8))` fails with `Undefined tensor b1_grad for reshape` even though `b1_grad` is a perfectly valid, populated tensor by that point in the program. This was actually the first workaround attempt for gotcha #3 above, and it failed for this unrelated, second reason — worth knowing these two are independent limitations, not the same one.

**5. `mse_loss` (and likely other losses) produce a `(1,1)` tensor, not a bare scalar. — not fixed, not a compiler bug**
Not a compiler bug exactly, but a real footgun for anyone driving TensorLang from Python: `float(np.load("loss.npy"))` throws in current numpy for a `(1,1)`-shaped array. Use `.item()` instead. Caught during the `tlkit` extraction (§13.1) precisely because a more realistically-shaped test fixture was used the second time around.

All five were originally noted in §10 "Known Pain Points" as candidates for an upstream fix; none were fixed in the compiler itself in this section's session. Four of the five (all but #5, which isn't a compiler bug) were fixed in a follow-up session — see §15.

---

## 14. Suggested Next Actions for a Returning Developer

1. Confirm environment: `bash build.sh --install` → activate venv → full test run.
2. Run `python3 tensorlang.py --app examples/hello_mlp` and inspect output.
3. Read `apps/examples/hello_mlp/main.tl` + `app.toml`.
4. ~~Run the full (unfiltered) `--no-parallel` suite once to close out §12.1 completely.~~ — **done, see §12.1**
5. ~~Add a `tests/` dir under `apps/examples/hello_mlp/` and run `--app examples/hello_mlp --test` to close out §12.2.~~ — **done, see §12.2** (also added for `linear_regression`)
6. ~~Pick one concrete practical application (image filter, tiny policy, portfolio predictor, …) and implement it as a second app under `apps/`.~~ — **done, see §13** (`decision_boundary`)
7. Log every friction point encountered; those become the real short-term language improvements. — **ongoing, see §13.3 for this round's five findings**
8. ~~Consider adding a quiet/default mode so app demos are less verbose.~~ — **done, see §12.3**
9. **New:** third app — GPU-batched physics playback (bouncing balls / double pendulum), built on `apps/tlkit/`. Pure `save()`-once-then-scrub, no chunking needed, and it's the first example to stress broadcasting across a batch dimension in a visual app — good odds of finding a sixth gotcha.
10. **New:** fourth app — interactive "click points, retrain, watch the boundary update" version of `decision_boundary`, triggered by a keypress instead of a fixed chunk count. Reuses the same `load()`/`save()`/subprocess pattern; sequence it after #9 so the chunked-runner plumbing is proven twice over before adding live interaction on top.
11. ~~Consider fixing the five §13.3 gotchas upstream in `type_checker.py`/`app_runner.py` rather than continuing to work around them app-side — especially #2 (silent exit-code-0 failures) and #3 (the 2D-vs-1D broadcast branch), which will keep costing debugging time on every future app that uses biases or any other 1D parameter.~~ — **done for #1/#2/#4, partially done for #3 — see §15**
12. **New:** add a `tests/` dir to `apps/examples/decision_boundary` (per the §12.2 pattern) once the network's converged output is stable enough to pin an `@EXPECTED` value against.
13. **New:** fix the CUDA kernel dispatch gap for `(1,1)`-scalar broadcasting found while closing out gotcha #3 — see §15.4. This is the one piece of §13.3 still open, and until it's fixed, decision_boundary and any future app must keep using the same-shaped-vector workaround rather than a true scalar learning rate.

---

## 15. Compiler Fixes (2026-09-06): Four of Five §13.3 Gotchas Resolved

A follow-up session went back through §13.3's five gotchas with the specific
goal of fixing them upstream instead of continuing to work around them
app-side. Four were fixed and verified end-to-end on real hardware (a
1080ti); the fifth turned out to be two bugs stacked on top of each other,
one fixed and one newly discovered and still open.

### 15.1 What was fixed

**Gotcha #1 (nested activation calls):** `tensorlang.lark`'s `relu_call`,
`gelu_call`, `swish_call`, `sigmoid_call`, and `tanh_call` now accept a full
`inline_expr` instead of a bare `NAME`, matching the pattern `matmul`/`add`/
`minus`/`mult`/`div` already had. `ast_builder.py`'s activation-call handler
was updated to parse the resulting `inline_expr` subtrees, and
`flatten_expr_args()` now also flattens these five ops, hoisting a nested
call into a synthetic `__tmp_N` binding exactly the way it already did for
binary ops. `tanh(matmul(x, w1))` now parses, flattens into
`__tmp_0 = matmul(x, w1)` / `h = tanh(__tmp_0)`, and computes the correct
numeric result — verified against a numpy reference on hardware (see
`tests/nested_activation_calls.tl`).

**Gotcha #2 (silent exit-0 on failed compile):** worse than originally
scoped — it wasn't only the type-check-failure path that was silent.
`compile_and_execute()` had five more early `return False, env` points
scattered through kernel generation (undefined save target, `load()` shape
mismatch, incompatible shapes during codegen) that were equally silent,
because none of the three call sites (`app_runner.py`'s normal-run path and
both benchmark loops, plus `tensorlang.py`'s CLI) ever checked the return
value. Fixed by: adding the missing `else` branch after type-checking
(prints a clear "Compilation aborted" message), normalizing every internal
early-exit point to `return False`, adding an explicit `return True` at the
actual point of successful completion, and updating all three callers to
check the return value and `sys.exit(1)` on failure. Verified on hardware:
a deliberately invalid program now exits `1` with a clear error, where it
previously exited `0` silently (see `tests/type_error_exit_code.tl`).

**Gotcha #4 (reshape can't take a `_grad` tensor):** turned out to affect
more ops than originally scoped — every single-tensor-field op had the same
gap, not just `reshape`/`transpose`/`max`/`min`/`argmax`/`argmin`/
`instance_norm`/`batch_norm` as originally listed. Fixed with a shared
`_resolve_tensor_type()` helper in `type_checker.py` carrying the same
`_grad`-suffix fallback `add`/`matmul`/etc. already had, applied to
`reshape`, `transpose`, `sum`/`mean`, `softmax`, `slice`, `layer_norm`,
`batch_norm`, `instance_norm`, `max`/`min`/`argmax`/`argmin`, `concat`, and
the activation ops. `reshape(w_grad, (8,))` now works directly — verified
on hardware, exact match against the expected all-ones gradient of `sum()`
(see `tests/reshape_gradient.tl`).

**Gotcha #3 (2D-vs-1D broadcast), type-checker half:** `type_checker.py`
now has an `_is_scalar_shape()` check (any shape where every dim is 1,
including `(1,1)`) checked before the old narrow special case, plus the
missing symmetric 1D-vs-2D case. A `(1,1)` scalar times an `(8,)` vector no
longer gets a false "shape mismatch" at the type-checking stage.

### 15.2 New tests

Three new files in `tests/`, plus a `test_runner.py` extension:

- `nested_activation_calls.tl` — gotcha #1, covers `tanh`/`sigmoid`/`relu`
  nested inside `matmul`/`add`, including the compound
  `relu(add(matmul(...), b))` case.
- `reshape_gradient.tl` — gotcha #4, `reshape(w_grad, ...)`.
- `type_error_exit_code.tl` — gotcha #2, a deliberately invalid program
  (matmul shape mismatch) that must exit non-zero.
- `test_runner.py` gained an `@EXPECT_FAILURE` marker
  (`is_expect_failure_test()` / an inverted pass condition in
  `run_single_test()`), since the harness previously had no way to assert
  "this program should correctly fail to compile" — every existing test
  only knew how to check a successful compile's output against an
  `@EXPECTED` block.

Test count: 106 → 109. All passing on hardware, confirmed via `bash
test.sh`.

### 15.3 Regression check

Before and after the fixes, the full `tests/*.tl` corpus was run through
parsing + AST building + type checking (no GPU needed) to check for
regressions: 104/106 pass on both, identical 2 pre-existing failures on
both (`load_dynamic.tl`, `load_inferred.tl` — missing cached `.npy` fixture
files, unrelated to these changes). Zero regressions from the fixes
themselves. The three new tests plus the full existing suite were then
verified end-to-end on real hardware (109/109).

### 15.4 New issue found, not fixed: CUDA kernel dispatch for scalar broadcast

Fixing the type-checker half of gotcha #3 exposed a second, deeper bug one
layer down. `compiler.py`'s codegen dispatch routes *any* `(2D, 1D)` shape
pair — including a disguised scalar like `(1,1)` — through
`kernel_generator.binary_broadcast()`, which unconditionally does
`rows, cols = output_shape`. Since the type-checker now assigns the
`(1,1) * (8,)` case an output shape of `(8,)` (1D), this line raises
`ValueError: not enough values to unpack` during kernel generation — before
ever touching the GPU. A purpose-built kernel for the true-scalar case
already exists (`binary_1d_broadcast`, docstring: "Handle broadcasting with
scalar (0-D tensor)"), but the dispatch only reaches it when *both*
operands are already 1D — never when the scalar is shaped `(1,1)`, which is
exactly the shape TensorLang's own losses/reductions (`mse_loss`, `sum`)
naturally produce. In other words: the real-world case that originally
motivated gotcha #3 — scaling a gradient by a loss or learning-rate scalar
— most likely still fails today, just later and louder (a Python
`ValueError`) instead of a clean type-check rejection.

Not fixed in this session: doing so properly means a dispatch branch that
detects a true scalar operand regardless of its declared rank, and a kernel
variant that preserves operand order for non-commutative ops (`minus`/
`div` — you can't just swap arguments into the existing scalar kernel,
since `A op B ≠ B op A`). That's new kernel-generation code, not a
one-line patch, and it needs a GPU to validate — left as an open item, see
§14.13. The existing app-side workaround (a same-shaped vector instead of a
true scalar, e.g. `lr_b1: Tensor[f32,(8,)]`) is still required until this
is fixed.

---

## 16. Contact / Context

Original author: davro (David)
Related projects of interest: `workspace` (PyQt6 IDE with Ollama), `we-not-me` (TradingView indicators).
Primary motivation for recent work: move from pure compiler infrastructure toward practical, visible applications.

---

*This handover assumes the state after the 2026-08-29 recovery, the 2026-09-03 test-runner/compiler-output fixes (§12), and the 2026-09-04 `tlkit`/`decision_boundary` session (§13). Update the "Current Health" and add a new dated session section when major changes land.*
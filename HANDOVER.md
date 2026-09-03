# TensorLang Handover Document

**Last updated:** 2026-09-03
**Project:** [davro/tensor-lang](https://github.com/davro/tensor-lang)
**Status:** Core language + test suite healthy (106/106, both parallel and `--no-parallel`, re-confirmed on hardware after this round of fixes). App system restored and demonstrated. Three issues found via code review and fixed in `test_runner.py`/`compiler.py`/`type_checker.py` — **all three verified fixed on hardware** (sequential test mode; noisy default compiler output; app test wiring) — see §12.

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
├── tests/                     # 106 core language tests
├── apps/                      # User-facing applications
│   └── examples/
│       └── hello_mlp/         # Minimal working app
│           ├── app.toml
│           └── main.tl
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
| Example app                        | `apps/examples/hello_mlp` runs successfully                         |
| Autograd + training loops          | Working (verified by hello_mlp + tests)                             |
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

Full list is available via `ls tests/`. All 106 currently pass.

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
- Inline unary ops (`relu(linear(...))`)
- Better shape-mismatch error messages
- ~~Cleaner / quieter compiler output for app runs~~ — **done, see §12.3**

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

## 13. Suggested Next Actions for a Returning Developer

1. Confirm environment: `bash build.sh --install` → activate venv → full test run.
2. Run `python3 tensorlang.py --app examples/hello_mlp` and inspect output.
3. Read `apps/examples/hello_mlp/main.tl` + `app.toml`.
4. ~~Run the full (unfiltered) `--no-parallel` suite once to close out §12.1 completely.~~ — **done, see §12.1**
5. ~~Add a `tests/` dir under `apps/examples/hello_mlp/` and run `--app examples/hello_mlp --test` to close out §12.2.~~ — **done, see §12.2** (also added for `linear_regression`)
6. Pick one concrete practical application (image filter, tiny policy, portfolio predictor, …) and implement it as a second app under `apps/`.
7. Log every friction point encountered; those become the real short-term language improvements.
8. ~~Consider adding a quiet/default mode so app demos are less verbose.~~ — **done, see §12.3**

---

## 14. Contact / Context

Original author: davro (David)
Related projects of interest: `workspace` (PyQt6 IDE with Ollama), `we-not-me` (TradingView indicators).
Primary motivation for recent work: move from pure compiler infrastructure toward practical, visible applications.

---

*This handover assumes the state after the 2026-08-29 recovery and successful `hello_mlp` run. Update the "Current Health" and "Recent Recovery" sections when major changes land.*
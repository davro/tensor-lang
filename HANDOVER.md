# TensorLang Handover Document

**Last updated:** 2026-09-27
**Project:** [davro/tensor-lang](https://github.com/davro/tensor-lang)
**Status:** Core language + test suite healthy (122/122). README updated to match implementation. Lint clean (`ruff check tensorlang/` passes). `build.sh --lint` now returns to the sourcing shell instead of exiting it. App system restored and demonstrated with two working apps (`hello_mlp`, `decision_boundary`). New shared toolkit (`apps/tlkit/`) for building visual/interactive apps. The five compiler/runner limitations discovered while building `decision_boundary` (§13.3) are now fixed upstream and hardware-verified (§15, §19). New app category `apps/games/` started (`tic_tac_toe` done; `2048` in progress). Building `2048`'s tensor-ops game engine (`step.tl`) surfaced and fixed four more real compiler bugs — two AST-inlining renaming gaps, one entirely missing CUDA kernel (`concat` axis=1), and one execution-ordering/priority bug in GPU-computed alias resolution — all hardware-verified; see §17. New (2026-09-26): a second physical GPU was added to the dev machine (2x GTX 1080 Ti); process-level ("Tier-A") multi-GPU pinning is now implemented for the test runner and single/app runs — see §18.

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
│   ├── app_runner.py          # Application runner
│   ├── ast_builder.py
│   ├── autograd.py
│   ├── compiler.py
│   ├── gpu.py                 # GPU detection + CUDA_VISIBLE_DEVICES pinning
│   ├── kernel_generator.py
│   ├── tensor_lang.py         # Argument parsing etc.
│   ├── tensor_verifier.py
│   ├── test_runner.py
│   └── type_checker.py
├── tensorlang.lark            # Grammar
├── build.sh                   # Install / test / lint helper
├── tests/                     # 122 core language tests
│   └── python/                # Plain-Python unittest suite
├── apps/                      # User-facing applications
│   ├── tlkit/                 # Shared toolkit for visual/interactive apps
│   │   ├── chunked_runner.py
│   │   ├── sequence_player.py
│   │   ├── colormap.py
│   │   └── README.md
│   ├── examples/
│   │   ├── hello_mlp/
│   │   └── decision_boundary/
│   └── games/
│       ├── tic_tac_toe/
│       └── 2048/
├── examples/                  # Older / misc examples
├── cache/                     # Generated CUDA, .npy, etc. (gitignored)
├── check.py                   # Parse + typecheck any .tl file, no GPU needed
├── verify.py                  # NumPy-interpret a .tl AST, no GPU needed
└── requirements.txt
```

---

## 4. Current Health

| Component | Status |
|---|---|
| Core test suite (default, parallel) | **122/122 passed** |
| Lint (`ruff check tensorlang/`) | **Clean** |
| `build.sh --lint` | Safe when sourced |
| AppRunner | Working for normal app execution |
| App test mode (`--app X --test`) | Working |
| Example apps | `hello_mlp`, `decision_boundary` verified |
| Shared app toolkit | `apps/tlkit/` available |
| Autograd + training loops | Working |
| Games app category | `tic_tac_toe` done; `2048` engine verified |

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
python3 tensorlang.py --list-apps
python3 tensorlang.py --app examples/hello_mlp
python3 tensorlang.py --app examples/decision_boundary
```

---

## 6. Application System (`apps/`)

Apps are discovered by the presence of an `app.toml`.

### Required layout

```
apps/
└── [<category>/]
    └── <app-name>/
        ├── app.toml
        ├── <entry>.tl
        └── tests/
            └── *.tl
```

### Minimal `app.toml`

```toml
[app]
name = "hello_mlp"
description = "Minimal 2-layer MLP training example"

[requirements]
gpus = 1
memory_gb = 4

[entry_points]
main = "main.tl"
```

### App requirements

* `requirements.gpus` is enforced via least-utilized GPU selection and process pinning.
* `requirements.packages` are auto-installed if missing.
* `cuda_version` and `memory_gb` are display-only.
* `[gpu] device = N` can pin an app to a specific GPU.

### App tests

```bash
python3 tensorlang.py --app examples/hello_mlp --test
```

App tests run sequentially.

---

## 7. Language Features

- Explicit tensor types: `Tensor[f32, (batch, features)]`
- 40+ operations (matmul, element-wise, activations, norms, losses, reductions, slicing, etc.)
- Reverse-mode autodiff (`with grad` + `backward()`)
- Training loops with weight rebinding (`w = w_updated`) — pointer swap, no copy
- Inline binary expressions (flattened to temporaries)
- User-defined functions
- Control flow in grammar only: `if` / `elif` / `else` is **not currently built or executed**
- File I/O (`load` / `save`)
- Caching of kernels and intermediate tensors

**Important constraint:**

All kernels from a compilation unit (including loop bodies) go into one `.cu` file. Tensor names used inside a `for` loop body cannot be reused at top level.

---

## 8. Test Inventory

The suite covers:

- Basics, arithmetic, broadcasting
- Activations (relu, sigmoid, tanh, softmax, gelu, swish)
- Reductions, slicing, reshape, transpose, concat
- Linear layers, MLP, neural nets
- Normalisations (layer/batch/instance)
- Losses (mse, cross-entropy)
- Autograd (basic, chain, broadcast, relu, sigmoid, tanh, weight update, linear, layer_norm, cross_entropy, gelu, swish)
- Training loops (`for_loop_*`, `training_pipeline`, `linear_regression`, …)
- Functions, pipelines, portfolio-style examples
- Load/save

Full list is available via `ls tests/`. All 122 currently pass.

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

### Short term

- Scoped kernel names (remove post-loop renaming constraint)
- Better shape-mismatch error messages
- Expand for-loop body operation coverage
- Alias resolution inside loop bodies
- Support `save()` inside loops
- Implement or remove `if`, `any()`, and `all()`

### Medium term

- Built-in optimisers (SGD, Adam) as language primitives
- Kernel fusion
- Buffer pooling for long training runs
- Nested loops
- Hot reload in dev mode
- Zero-copy Python/PyTorch interop

### Long term

- Multi-GPU single-program execution (Tier-B)
- Mixed precision (FP16/FP32), validated on appropriate hardware
- MLIR integration
- Additional hardware backends (ROCm, Metal)

### Product / motivation

- More practical example applications (vision, simple game agent, signal processing, finance) so the language does not feel purely abstract.

---

## 11. Recent Recovery Notes (Aug 2026)

- Fresh clone + `bash build.sh --install` succeeded.
- `ModuleNotFoundError: tensorlang.app_runner` occurred because the file was missing from main.
- File was restored from prior local work; import re-enabled.
- Full test suite went green.
- First app (`examples/hello_mlp`) runs end-to-end and matches documented training numbers.

---

## 12. Known Issues & Recent Fixes (2026-09-03)

- `--no-parallel` crashed with a `TypeError` — fixed and verified.
- `--app <name> --test` crashed with a `TypeError` — fixed and verified.
- Default compiler output was extremely noisy — fixed and verified.
- `app_runner.py` was absent from main; restored from previous work.

---

## 13. Session 2026-09-04: `apps/tlkit/`, second app (`decision_boundary`), and compiler limitations

### 13.1 `apps/tlkit/`

Shared toolkit for visual/interactive apps:

- `chunked_runner.py`
- `sequence_player.py`
- `colormap.py`

### 13.2 `apps/examples/decision_boundary`

Trains a 2-8-1 tanh/sigmoid MLP on XOR and animates the decision boundary. Uses chunked process invocations because `save()` targets are fixed string literals and rebind requires fixed-shape GPU buffers.

Verified result: loss dropped from 0.243585 to 0.001573 over 1000 epochs.

### 13.3 Limitations found while building it

1. Activation functions only accepted bare `NAME` arguments — fixed.
2. `app_runner.py` did not propagate compile/execute failure into exit code — fixed.
3. `mult`/`add`/`minus`/`div` 2D-vs-1D broadcast branch was too narrow — fixed.
4. `reshape()` could not take a gradient tensor — fixed for all single-tensor-field ops.
5. `mse_loss` produces a `(1,1)` tensor, not a bare scalar — not a compiler bug; use `.item()` in Python.

---

## 14. Suggested Next Actions

1. Confirm environment and run the full test suite.
2. Read `apps/examples/hello_mlp/main.tl` + `app.toml`.
3. Add tests under apps as needed.
4. Log every friction point encountered.
5. Finish `apps/games/2048`.
6. Consider generalizing `build_ast()` function-call inlining to arbitrary nesting depth.

---

## 15. Compiler Fixes (2026-09-06)

### 15.1 What was fixed

- Nested activation calls
- Silent exit-0 on failed compile
- Gradient tensor support for single-tensor-field ops
- 2D-vs-1D broadcast type-checker half

### 15.2 New tests

- `nested_activation_calls.tl`
- `reshape_gradient.tl`
- `type_error_exit_code.tl`
- `@EXPECT_FAILURE` support in `test_runner.py`

Test count: 106 → 109.

### 15.3 Regression check

No regressions from the fixes themselves. Full suite verified on hardware.

### 15.4 Fixed: CUDA kernel dispatch for scalar broadcast

Both halves of the scalar-broadcast issue are fixed. `compiler.py` detects true scalar operands regardless of declared rank and routes them to `binary_1d_broadcast`, preserving operand order for non-commutative ops.

---

## 16. Contact / Context

Original author: davro (David)  
Related projects of interest: `workspace` (PyQt6 IDE with Ollama), `we-not-me` (TradingView indicators).  
Primary motivation for recent work: move from pure compiler infrastructure toward practical, visible applications.

---

## 17. 2048 App: Tensor-Ops Game Engine + Four Compiler Fixes (2026-09-08/09)

A new app category, `apps/games/`, was started with `tic_tac_toe` and `2048`.

`2048` implements game mechanics as tensor arithmetic in `apps/games/2048/step.tl`.

### 17.1 Four real compiler bugs found and fixed

1. `substitute_names()` did not rename `concat` operands.
2. Nested function-call renaming had the same gap for `tensors` and `tensor`.
3. `concat(axis=1)` had no CUDA kernel.
4. GPU-computed function-return aliases resolved incorrectly.

### 17.2 New CUDA kernel: `concat_axis1`

A genuinely new, previously missing kernel.

### 17.3 New general-purpose dev tooling

- `check.py` — parse + typecheck without GPU
- `verify.py` — NumPy interpreter for AST validation

### 17.4 Verification

- `check.py` passes
- `verify.py` passes 6/6 cases
- Hardware verification passes for single-row and full-board cases

### 17.5 Suggested next actions

1. Right/up/down moves
2. `agent.py`
3. `generate_data.py`
4. Train/infer/play pipeline
5. Score tracking
6. Consider fixing recursive inlining

---

## 18. Multi-GPU Tier-A Support (2026-09-26)

### 18.1 `tensorlang/gpu.py`

- `detect_gpus()`
- `pin_env()`
- `pin_current_process()`
- `choose_device()`

`choose_device()` is now wired into `AppRunner._validate_requirements()`. `requirements.gpus` is enforced via least-utilized GPU selection and process pinning.

### 18.2 `test_runner.py`

Round-robin GPU pinning across subprocess tests.

### 18.3 Measured results

Round-robin pinning is real but modest on the small/fast correctness suite. Heavier concurrent GPU workloads are the better use case.

### 18.4 Suggested next actions

1. `requirements.gpus` is now enforced; `requirements.packages` is auto-installed; `cuda_version` and `memory_gb` remain display-only. A small `app.toml` schema addition is still useful for apps that want a specific card.
2. Measure Tier-A on heavier workloads.
3. Tier-B remains a future, larger change.

---

## 19. Audit + Lint/Autograd Coverage Pass

A full codebase audit was performed across:

- `tensorlang.lark`
- `tensorlang/ast_builder.py`
- `tensorlang/type_checker.py`
- `tensorlang/autograd.py`
- `tensorlang/kernel_generator.py`
- `tensorlang/compiler.py`
- `tensorlang/test_runner.py`
- `tensorlang/app_runner.py`
- `tensorlang/gpu.py`
- `tensorlang/tensor_verifier.py`
- `tensorlang/tensor_lang.py`

### Fixes applied

1. Broadcast `equal` dispatch
2. `(1,1)` scalar broadcast dispatch
3. `transpose(..., axes=...)`
4. `layer_norm(..., axis=0)`
5. `linear` autograd wiring
6. `layer_norm` autograd wiring
7. Full `argmax` / `argmin` reductions
8. `gelu` and `swish` forward + autograd
9. `cross_entropy` autograd wiring
10. `build.sh --lint` graceful handling
11. Lint cleanup across the codebase

### Verification

- `ruff check tensorlang/` — **All checks passed**
- `source build.sh` — **122/122 tests passed**

### Remaining limitations

- `if`, `any()`, `all()` are grammar-only
- `f64` is parsed but effectively `f32`
- For-loop bodies support only a subset of operations
- Most kernels are 1D/2D only
- `batch_norm` has no gamma/beta
- Dev-mode hot reload is not implemented
- `--transpile` (WGSL) is not functional
- Tier-B multi-GPU single-program execution is not implemented

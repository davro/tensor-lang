# TensorLang: A Native AI/ML Language

TensorLang is an open-source, native machine-learning language designed to compile tensor operations into GPU-accelerated CUDA kernels.

This project addresses pain points in Python-based ML frameworks by offering a streamlined, hardware-aware alternative for high-performance tensor computations. The compiler is implemented in Python, but the hot path — tensor operations — runs as compiled CUDA kernels.

> **Status:** v0.5.10 released (training that scales: data-parallel training across GPUs with `--ddp`, a device buffer pool and a leaner `backward()` for the GPU backward, and the tools that measured them). v0.5.9 added `softmax_cross_entropy`, an Adam precision fix and the opt-in GPU backward for `matmul` and `gelu`; v0.5.8 added `embedding` and real GPT-2 small, whose logits match a float64 NumPy reference to 1.1e-4. Core suite: 319/319 `.tl` tests passing; 172 Python unit tests.
> **Hardware validated on:** 2× NVIDIA GTX 1080 Ti (Pascal, sm_61).
> **Current focus:** v0.5.11 — training that stays on the GPU (gradients and activations resident, no round trip per op, then NCCL for data-parallel runs), a data-parallel check on a transformer (Adam, layer norm, softmax, embedding), a multi-head training program for GPT-2 small, and `load()` indexed by the loop counter so a loop can see different batches; then the earlier list (generation inside the language, `where`, `fill` as an operand, a diagnostic for a redefined `let` name). See [Future Work](#future-work); earlier releases are in [RELEASES.md](RELEASES.md).

---

## Table of Contents

* [Overview](#overview)
* [Current Status and Limitations](#current-status-and-limitations)
* [Setup](#setup)
* [Quick Start](#quick-start)
* [Documentation](#documentation)
* [Repository Layout](#repository-layout)
* [Language Overview](#language-overview)
* [Operations](#operations)
* [Control Flow](#control-flow)
* [Automatic Differentiation](#automatic-differentiation)
* [Mixed Precision](#mixed-precision)
* [Training Loops](#training-loops)
* [Optimizers](#optimizers)
* [Data-Parallel Training](#data-parallel-training)
* [File I/O](#file-io)
* [NumPy Interop](#numpy-interop)
* [Cache](#cache)
* [Diagnostics](#diagnostics)
* [Architecture](#architecture)
* [Performance](#performance)
* [Testing](#testing)
* [Applications](#applications)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [About](#about)
* [License](#license)

---

## Overview

TensorLang provides a unified stack for parsing, type checking, CUDA kernel generation, compilation, and execution.

**Key Benefits:**

* **Native GPU acceleration** — tensor operations compile to CUDA kernels
* **cuBLAS-accelerated matmul** — matrix multiplication dispatches to cuBLAS for performance, including batched N-D
* **Type safety** — compile-time tensor shape and broadcasting checks (N-D aware)
* **Stable diagnostics** — type-checker errors carry stable codes (e.g. `[E-SHAPE-MISMATCH]`) so tools and tests can match on the code while the wording improves
* **ML-first design** — built specifically for neural network operations
* **Automatic differentiation** — reverse-mode autograd for a broad set of operations (FP32 and FP16)
* **Control flow** — scalar and tensor-valued `if` / `elif` / `else`, with branch shape/dtype consistency checks
* **Training loops** — `for` loops with `backward()`, pointer-swap weight rebinding, nested loops, and `save()` inside loops
* **Built-in optimizers** — first-class `sgd` and `adam` with compiler-managed state, FP32 master weights for FP16 params
* **Elementwise kernel fusion** — fuses elementwise chains (e.g. `add → relu`), including broadcasting cases, into a single CUDA kernel
* **Inline expressions** — nested binary/unary calls are flattened into synthetic bindings
* **Scoped kernel names** — loop-local CUDA symbols prevent name collisions when tensor names are reused
* **NumPy zero-copy interop** — share CPU buffers between TensorLang tensors and NumPy arrays without copying
* **Mixed precision (FP16/FP32)** — FP16 storage with FP32 accumulation; cuBLAS tensor-core matmul path; cast-based elementwise/activation/reduction kernels
* **N-dimensional tensor support** — general stride-based indexing across elementwise, reductions, shape ops, normalisations, activations, and batched matmul
* **Loss scaling** — static and dynamic loss scaling for FP16 training stability; auto-detect or explicit scale via pragma; gradients upcast to FP32 when scale overflows FP16 max; overflow detection with automatic scale reduction
* **FP16 master weights** — Adam/SGD with FP32 master weights for FP16 params; optimizer computes in FP32, casts to FP16 for forward pass
* **Extensible architecture** — clean separation of parsing, type checking, and code generation (codegen, execution, interop, and per-category kernel generators)

**Important framing:** Python is the host/compiler language. Tensor operations run as native CUDA kernels, but the compiler, runtime orchestration, and test runner are Python-based.

---

## Current Status and Limitations

TensorLang is usable for small-to-medium static-shape ML programs, but several features are intentionally limited or not yet implemented.

### Implemented and working

* Tensor declarations with explicit shapes: `Tensor[f32, (batch, features)]`, `Tensor[f16, (batch, features)]`, and N-D: `Tensor[f32, (batch, channels, height, width)]`
* Element-wise arithmetic with broadcasting (N-D aware, NumPy-style)
* Matrix multiplication (`matmul`) — cuBLAS-backed (FP32 and FP16 tensor core path), including batched N-D
* Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish` (N-D aware)
* Reductions: `sum`, `mean`, `min`, `max`, `argmin`, `argmax` (N-D aware, arbitrary axis)
* Comparisons: `greater`, `less`, `equal`, `greater_equal`, `less_equal`, `not_equal`, also written as the operators `>`, `<`, `==`, `>=`, `<=`, `!=` (equal shapes, or a `(rows, cols)` tensor against a `(cols,)` row; see Known limitations)
* Condition comparisons: `>`, `<`, `==`, `>=`, `<=`, `!=` (output always f32); a numeric literal takes the shape of the tensor it is compared with
* Slicing, reshape, transpose, concat (N-D aware)
* Normalisations: `layer_norm`, `batch_norm`, `instance_norm` (N-D aware)
* Losses: `mse_loss`, `cross_entropy` (N-D aware)
* Linear layers: `linear` (N-D aware)
* Reverse-mode autograd for a broad set of operations (FP32 and FP16), including `batch_norm` (with `gamma`/`beta`) and `instance_norm` (simplified gradients)
* Multiple `backward()` statements in one file; a pass only overwrites the gradients of parameters its loss depends on (see [Automatic Differentiation](#automatic-differentiation))
* Training loops with weight rebinding, nested loops, `save()` inside loops, and alias resolution
* Control flow: scalar/single-element and tensor-valued `if` / `elif` / `else` (grammar, AST, and execution); conditions may compare against numeric literals (`x > 0.5`); branch shape/dtype consistency is enforced
* Built-in optimizers: `sgd`, `adam` (compiler-managed state, multi-parameter, FP32 master weights for FP16 params)
* Elementwise kernel fusion for elementwise chains, including broadcasting (row/column/general broadcast modes, N-D aware)
* Scoped kernel names for loop bodies
* User-defined functions, including calls inside `for` loop bodies
* File I/O: `load()` (including inside `for` loop bodies, f32 and f16 files), `save()`
* NumPy zero-copy interop (CPU-side buffer sharing, float16/32/64)
* Comments: `//` and `/* */`
* Application system (`apps/`) with `hello_mlp`, `decision_boundary`, `tic_tac_toe`, and `2048` (engine verified)
* Shared app toolkit (`apps/tlkit/`)
* **Mixed precision (FP16/FP32)** — FP16 storage with FP32 accumulation; cast-based kernels; cuBLAS tensor-core matmul path; f16 autograd and training loops
* **N-dimensional tensor support** — general stride-based indexing across elementwise, reductions, shape ops, normalisations, activations, and batched matmul
* **Autograd coverage expansion** — gradients wired for `reshape`, `transpose`, `slice`, `concat`, `max`, `min`, `argmax`, `argmin`, comparisons (zero gradient for non-differentiable ops), `batch_norm`, and `instance_norm`
* **Loss scaling** — static and dynamic loss scaling for FP16 training stability; configurable via `# loss_scale: <value|auto|dynamic>` pragma; gradients upcast to FP32 when scale overflows FP16 max; overflow detection with automatic scale reduction (halve on overflow, double after N successful steps); unscale before optimizer step
* **FP16 master weights** — Adam/SGD with FP32 master weights for FP16 params; optimizer computes in FP32 with FP32 m/v buffers, casts to FP16 for forward pass; automatic for FP16 params with grad
* **Batched matmul gradients** — N-D (batched) matmul backward via NumPy batched matmul + `_unbroadcast` for broadcast dims
* **For-loop body parity** — normalisations, comparisons, slicing, reshape, transpose, concat, additional reductions, user-defined functions, and `load()` execute in loop bodies
* **`any()` / `all()` reductions** — full and axis reductions, N-D aware, f16 support, zero gradients for autograd
* **Tensor-valued control flow** — `if` / `elif` / `else` conditions may be multi-element tensors; truthiness is all-nonzero
* **Comparison operators as expressions** — `let m = x >= y`, `let m = x != 2.0` (lowered to the named operations; chained comparisons such as `a > b > c` are a syntax error)
* **`backward()` inside branches** — in any `if`, `elif` or `else` branch, including nested branches (two levels and deeper); each `backward()` runs where it appears, several may appear in one branch, and the loss may depend on values computed before the branch or in an enclosing branch
* **Stable diagnostic codes** — see [Diagnostics](#diagnostics)
* **Transformer building blocks** — a one-layer pre-LN GPT-2-style forward pass (token and positional embedding, layer norm, causal attention, residual connections, gelu FFN, final layer norm, output projection) is written with existing operations only, with no attention primitive, at toy size (seq 4, d 4, vocab 8). With seeded distinct weights and a real one-hot input it matches a float64 NumPy reference on the GPU (`probe_gpt2_forward`, generated by `tools/make_gpt2_probe.py`, whose built-in check confirms that a wrong mask, scale, residual, bias placement or activation changes the logits). Attention gradients match NumPy (`probe_attention_block`). The causal mask is built without a dedicated op: `(rows, 1) - (1, cols)` broadcast, `>= 0.0`, then an additive `-1e9`
* **`embedding` and real GPT-2** — `embedding(indices, weight)` gathers rows (gradient: scatter-add into the weight). With it, the real GPT-2 small (12 layers, 12 heads, 124M parameters) runs from a generated TensorLang program built from existing operations, and matches a float64 NumPy reference (max logit difference 1.1e-4 on logits of magnitude ~83; identical argmax and top-5). See [Running GPT-2](docs/gpt2_demo.md).

* **Data-parallel training (`--ddp`)** — one worker per GPU on the same program; gradients are averaged after every `backward()` and the overflow flag is shared, so workers take identical steps; rank 0's parameters are broadcast at start-up; `{rank}` / `{world_size}` placeholders in `load()` and `save()` paths; a shared kernel build and per-rank output directories. See [Data-Parallel Training](#data-parallel-training)
* **GPU backward buffer pool and a leaner `backward()`** — the opt-in GPU backward (`TENSORLANG_GPU_BACKWARD=1`) reuses its device buffers (`TENSORLANG_GPU_BACKWARD_POOL=0` turns that off), and the autograd pass zero-fills, tests for zero gradients and checks for overflow with cheaper calls, with bit-identical results

### Known limitations

* **Control flow:** `if` / `elif` / `else` are implemented with defined condition semantics (see [Control Flow](#control-flow)) and branch shape/dtype consistency is enforced. A `for` loop body or a branch may contain several `backward()` statements; each runs where it appears.
* **Comparison operands:** the CUDA comparison kernels run two operands of equal shape (any rank), or a `(rows, cols)` operand against a `(cols,)` or `(1, cols)` row. Any other combination (for example `(3,)` against `(1,)`, or a row against a matrix) is a compile error, `[E-SHAPE-MISMATCH]`, for comparison operations and for `if`/`elif` conditions. f16 operands are a compile error too, `[E-COMPARISON-DTYPE]`, because the comparison kernels are always generated for f32; that includes an f16 tensor compared with a numeric literal, which is an f32 fill. A numeric literal takes the shape of the tensor it is compared with. `tools/scan_comparison_shapes.py` and `tools/scan_comparison_dtypes.py` list the comparisons in a codebase that these rules would affect.
* **Aliases track their source:** a top-level alias (`let a = z`) is another name for `z` and reflects any later rebind, including rebinds inside branches. It is not a snapshot.
* **Multiple `backward()` statements:** each pass updates the gradients of the parameters its loss depends on; other parameters keep the gradient from the last pass that reached them, and a parameter no pass has reached has a zero gradient. There is no implicit gradient zeroing (see [Automatic Differentiation](#automatic-differentiation)). The interaction with loss scaling is not yet tested.
* **`f64`:** a tensor declared as `f64` is a compile error, `[E-UNSUPPORTED-DTYPE]`. The grammar parses `f64`, but there is no f64 code generation: operations failed with a `KeyError`, and comparisons read the data as f32. Declare tensors as `f32`; the data in a `.npy` file can stay float64, because `load()` converts it to the declared type. An undeclared `load()` of a float64 `.npy` or `.npz` is inferred as f32 and converted on load (`--debug-info` prints a note). Full f64 support is not planned.
* **For-loop bodies:** operation parity with top level, including user-defined functions and `load()`. Loop counts must still be integer literals.
* **Axis support is limited** for several normalisations.
* **`batch_norm`** normalises with the *given* `running_mean`/`running_var`: it does not compute batch statistics or update running statistics (training-mode batch norm is not implemented). `gamma`/`beta` are optional, both or neither. The channel is dimension 1 (NCHW); the running statistics, `gamma` and `beta` must be f32 (`[E-BATCH-NORM-DTYPE]`) with shape `(C,)`.
* **`instance_norm`** N-D uses simplified spatial-dim flattening.
* **Growing a tensor in a loop:** a rebind whose new value has a different shape (for example `tokens = concat(tokens, new)`) is skipped with a diagnostic at run time. The type checker does not check rebinds, so the program type-checks. Autoregressive generation therefore runs on the host: Python drives the loop and the `.tl` program computes one forward pass. Test: `probe_growing_concat_loop`.
* **`embedding` indices are floats:** there is no integer dtype, so the indices are `f32` (exact below 2^24) and `f16` indices are a compile error, `[E-EMBEDDING-DTYPE]`. Ids are checked on the host before each launch (`[E-EMBEDDING-INDEX]`).
* **`load()` reads one array per file:** `.npy` is one array; for `.npz` it takes the array named `data`, else the first. The filename must be a string literal, so a model with many layers is generated and unrolled rather than looped over a path.
* **GPT-2 generation is a demonstration:** the loop runs on the host, one run of `tensorlang.py` per token, with no KV cache. Every step recomputes the whole sequence and reloads the weights (about 2.9 s per token on a GTX 1080 Ti once compiled; compiling a new sequence length costs about 16–18 s).
* **`fill` takes one scalar:** `fill(value, shape)` gives a constant tensor. For varied values use tensor literals (1-D or 2-D) or `load()`. `fill` is not accepted as an operand inside another call (`add(fill(...), x)` is a syntax error); bind it with its own `let`.
* **Redefining a name:** defining the same name twice with `let` passes type checking. The compiler then uses the shape from the *last* definition, so a redefinition with a different shape fails at compile time (for example `matmul inner dimension mismatch`), and a redefinition with the same shape is not a supported pattern. Use distinct names; mutation is what rebinds are for. A diagnostic for this is proposed (see [Future Work](#future-work)).
* **Constant test inputs hide errors:** layer norm of constant rows is zero (so everything after it is zero) and a softmax over constant values is uniform, so a probe built from constant fills can give all-zero or all-equal output whether or not a mask, scale or matmul is right. A first forward-pass probe with constant-per-column weights returned logits of about 1e-8 (roundoff around an exact zero) in both TensorLang and NumPy, and "agreement" there said nothing: removing the mask, changing the scale or replacing the attention output left it unchanged. A broadcast bug was hidden the same way: a 1-D bias or layer-norm gamma/beta on a square matrix was applied per row instead of per column, and constant or zero parameters (gamma = 1, bias = 0) gave the same answer either way, until a probe with distinct values exposed it (fixed in v0.5.7). Use distinct values (literals or `load()`), real one-hot inputs, and check that a deliberately broken block changes the result.
* **Dev mode hot reload** is not implemented.
* **`--transpile` (WGSL)** is exposed but not currently functional.
* **NumPy interop is CPU-side only** — GPU-side zero-copy requires CUDA unified memory (deferred).
* **Autograd N-D limitations** — `instance_norm` gradients are simplified. `batch_norm` gradients are exact for its forward (given statistics): `dx`, `dgamma` and `dbeta`; nothing flows to the running statistics.
* **Dynamic loss scaling** — overflow detection works but scale growth interval defaults to 2000 steps; may need tuning for short training runs.
* **Data-parallel training (`--ddp`)** is one machine, host memory over TCP (no NCCL), POSIX only. Shards must be the same size and the loss a mean, or the averaged gradient is not the whole-batch gradient. It has been checked on a small GELU network with SGD, not yet on a transformer with Adam, and there are no explicit collectives in the language yet. See [docs/data_parallel.md](docs/data_parallel.md).
* **The CPU `gelu` backward is slow:** the NumPy path calls `math.erf` per element (about 13 ms for 32K elements, 34 times the GPU path). It matters for runs without `TENSORLANG_GPU_BACKWARD=1` and for operands the GPU backward does not take.
* **Absolute program paths:** the compiler builds its cache directory as `cache/<path>`, so a program given by an absolute path fails with `File exists`; use a path relative to the repository.

---

## Setup

### System Requirements

```bash
sudo apt install nvidia-cuda-toolkit python3-dev
sudo apt install python3.12-venv
```

### Clone

```bash
git clone https://github.com/davro/tensor-lang.git
cd tensor-lang
```

### Build

```bash
chmod +x build.sh

# Install dependencies (run with bash, NOT source — see note below)
bash build.sh --install

# Activate the venv
source python-env/bin/activate

# Run the full test suite
source build.sh
```

**Important notes**

* Always prefer `bash build.sh --install`. Using `source build.sh --install` hits an `exit 0` that kills the current shell.
* `bash build.sh --test` is a safe way to run the full suite: it clears `cache/` first, then runs the `.tl` suite and the Python tests.
* `pycuda` requires a working NVIDIA CUDA toolkit.
* After install, the venv lives in `python-env/`.

---

## Quick Start

```bash
# Compile and run a single file
python3 tensorlang.py tests/activation_pipeline.tl

# Debug mode
python3 tensorlang.py tests/inline_expr_basic.tl --debug

# Run with caching of kernels and intermediate tensors
python3 tensorlang.py --cache-layers tests/training_pipeline.tl

# Run tests
python3 tensorlang.py --test
python3 tensorlang.py --test --filter autograd
```

---

## Documentation

The README is the language reference. Longer guides live in `docs/`.

| Guide | Covers |
|---|---|
| [docs/training.md](docs/training.md) | Training loops, choosing a loss (`softmax_cross_entropy` vs `cross_entropy`), optimizers, the opt-in GPU backward, profiling, and how a training program is verified |
| [docs/gpt2_demo.md](docs/gpt2_demo.md) | Running real GPT-2 small: requirements, the steps, what each one writes, timings and what was verified |
| [docs/weight_space_programming.md](docs/weight_space_programming.md) | Editing GPT-2's weights and running the edited model, with the experiments and their limits |
| [docs/data_parallel.md](docs/data_parallel.md) | Data-parallel training with `--ddp`: what is synchronised, `{rank}` placeholders, the launcher and its environment, cache layout, messages, limits |
| [docs/handovers/HANDOVER_MULTI_GPU.md](docs/handovers/HANDOVER_MULTI_GPU.md) | Design notes and implementation status of the multi-GPU work |
| [docs/apps.md](docs/apps.md) | Writing an application: directory layout, `app.toml`, requirements, and app tests |
| [docs/mixed_precision.md](docs/mixed_precision.md) | Loss scaling (static, auto, dynamic), FP32 master weights for f16 parameters, and how they combine in training loops |
| [docs/testing.md](docs/testing.md) | Test annotations (`@EXPECTED`, `@EXPECT_FAILURE`), tensor verification, what the suite covers, helper tools, multi-GPU test runs |
| [RELEASES.md](RELEASES.md) | Release history, newest first: what each version added, found and fixed |
| [docs/social/0.5.9/RELEASE_NOTES_v0.5.9.md](docs/social/0.5.9/RELEASE_NOTES_v0.5.9.md) | What changed in v0.5.9, with the measurements |

## Repository Layout

```
tensor-lang/
├── tensorlang.py              # CLI entry point
├── tensorlang/                # Core package
│   ├── app_runner.py          # Application discovery and execution
│   ├── ast_builder.py         # AST construction and inlining
│   ├── autograd.py            # NumPy reference autograd (scale + overflow detection)
│   ├── codegen.py             # Code generation orchestration
│   ├── compiler.py            # Orchestration, execution, caching, scoped kernel names, loss scale
│   ├── constants.py           # Shared constants (dtype registry, promotion, loss scale)
│   ├── dist.py                # Data-parallel collectives (host-side TCP) and gradient averaging
│   ├── execution.py           # Runtime execution, loops, rebinding
│   ├── fusion.py              # Elementwise kernel fusion (broadcasting-aware)
│   ├── generators/            # Per-category CUDA kernel generators
│   │   ├── base.py            # Shared helpers, dtype cast templates
│   │   ├── comparisons.py     # greater, less, equal
│   │   ├── elementwise.py     # add, minus, mult, div, fill, fused
│   │   ├── nn.py             # matmul, linear, norms, losses
│   │   ├── optimizers.py     # sgd_update, adam_update (master weights)
│   │   ├── reductions.py     # sum, mean, min, max, argmin, argmax, softmax
│   │   ├── shapes.py         # reshape, transpose, slice, concat
│   │   └── unary.py          # relu, sigmoid, tanh, gelu, swish
│   ├── gpu.py                 # GPU detection + CUDA_VISIBLE_DEVICES pinning
│   ├── gpu_backward.py        # Opt-in GPU backward for matmul and gelu (device buffer pool)
│   ├── interop.py             # NumPy zero-copy interop
│   ├── io.py                  # File I/O helpers
│   ├── kernel_cache.py        # Shared kernel build and per-rank cache directories (--ddp)
│   ├── kernel_generator.py    # CUDA kernel generation
│   ├── launcher.py            # --ddp launcher: one worker per GPU
│   ├── tensor_lang.py         # Argument parsing etc.
│   ├── tensor_verifier.py     # Tensor verification helpers
│   ├── test_runner.py         # Test discovery and execution
│   └── type_checker.py        # Shape and broadcasting checks
├── tensorlang.lark            # Grammar
├── build.sh                   # Install / test / lint helper
├── tests/                     # Core language tests (.tl) and small data fixtures (.npy)
│   └── python/                # Plain-Python unittest suite
├── docs/                      # Guides (see Documentation) and handover notes
├── tools/                     # Profilers, generators and checks (see docs/training.md)
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
└── requirements.txt
```

---

## Language Overview

### Tensor Declarations

```text
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

// Mixed precision: FP16 storage
let x16: Tensor[f16, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]

// N-dimensional: 4D tensor
let img: Tensor[f32, (batch, channels, height, width)] = ...
```

Supported dtypes: `f16`, `f32`. (`f64` is parsed but rejected with `[E-UNSUPPORTED-DTYPE]`.)

There are no scalar values. A bare number cannot be bound with `let` (`[E-SCALAR-LITERAL]`). Numeric literals are valid only in comparisons (`if x > 0.5`, `let m = x >= 0.5`), in optimizer arguments (`sgd(w, w_grad, 0.01)`), and as `fill` values. For a constant operand, use a `(1,)` tensor:

```text
let half: Tensor[f32, (1,)] = [0.5]
let y = mult(x, half)
```

Shapes are part of the type system. The compiler checks shape compatibility and broadcasting rules at compile time, including N-D broadcasting (NumPy-style: align trailing dims, broadcast size-1 dims).

### User-Defined Functions

```text
fn normalize(x: Tensor[M, N]) -> Tensor[M, N] {
    let mu = mean(x, axis=1)
    return minus(x, mu)
}
```

Function calls are inlined, including calls inside `for` loop bodies. Generic dimensions such as `M` and `N` are primarily syntactic; concrete shapes are resolved at call sites.

### Inline Expressions

Nested binary and unary calls are flattened into synthetic `__tmp_N` bindings:

```text
let w_updated = minus(w, mult(lr, w_grad))
```

This is equivalent to:

```text
let __tmp_0 = mult(lr, w_grad)
let w_updated = minus(w, __tmp_0)
```

### Aliases

A top-level `let a = z` makes `a` another name for `z`. After execution the compiler re-resolves aliases, so `a` reflects any later rebind of `z` (including a rebind inside a branch). If you need an independent copy, use an explicit operation rather than an alias.

---

## Operations

### Tensor Creation and Manipulation

| Operation | Status | Notes |
|---|---|---|
| Tensor literals | Supported | `[[1.0, 2.0], [3.0, 4.0]]` |
| `fill(value, shape)` | Supported | Scalar fill value only |
| Slicing | Supported | N-D, NumPy-style ranges and indices (index removes dim, slice keeps dim) |
| `reshape(tensor, shape)` | Supported | Element count must match; N-D |
| `transpose(tensor, axes=...)` | Supported | General N-dimensional; axes are validated |
| `concat(a, b, axis=...)` | Supported | 2 tensors, any N-D, any axis |
| `embedding(indices, weight)` | Supported | Gathers rows: `weight` is `(vocab, d)`, `indices` is `f32` of any rank >= 1; the result is `indices.shape + (d,)` in the weight's dtype. Out-of-range or non-integer ids are a run-time error, `[E-EMBEDDING-INDEX]` |

### Linear Algebra

| Operation | Status | Notes |
|---|---|---|
| `matmul(A, B)` | Supported | N-D batched: (..., m, k) @ (..., k, n) → (..., m, n); cuBLAS-backed (FP32 and FP16 tensor core path) |
| `add`, `minus`, `mult`, `div` | Supported | N-D broadcasting (NumPy-style); dtype-aware (f16/f32) |
| `linear(input, weight, bias)` | Supported | N-D: operates on last dim; dtype-aware |

### Activations

| Operation | Status | Gradient |
|---|---|---|
| `relu` | Supported | Yes |
| `sigmoid` | Supported | Yes |
| `tanh` | Supported | Yes |
| `softmax` | Supported | Yes (any axis, N-D) |
| `gelu` | Supported | Yes |
| `swish` | Supported | Yes |

### Reductions

| Operation | Status | Notes |
|---|---|---|
| `sum` | Supported | Full or any axis; N-D; accumulates in f32 |
| `mean` | Supported | Full or any axis; N-D; accumulates in f32 |
| `min`, `max` | Supported | Full or any axis; N-D |
| `argmin` | Supported | Full reduction or any axis; N-D |
| `argmax` | Supported | Full reduction or any axis; N-D |

### Normalisations

| Operation | Status | Notes |
|---|---|---|
| `layer_norm` | Supported | N-D, any axis |
| `batch_norm` | Supported | `batch_norm(x, mean, var [, gamma, beta] [, eps=...])`, N-D (channel at dim 1, NCHW); given running statistics, optional affine `gamma`/`beta`; no training-mode statistics |
| `instance_norm` | Partial | N-D (per-instance over non-batch dims); simplified gradient |

### Losses

| Operation | Status | Notes |
|---|---|---|
| `mse_loss` | Supported | N-D; predictions and targets must have the same shape. Result is a `(1,1)` tensor, not a bare scalar — use `.item()` in Python. |
| `cross_entropy` | Supported | N-D; probability inputs and class-index targets (last dim = classes) |
| `softmax_cross_entropy(logits, targets)` | Supported | Fused, stable loss on logits (f32 or f16, N-D, last dim = classes); targets are `f32` class indices of shape `logits.shape[:-1]`; result is an f32 scalar, the mean over rows. Gradient `(softmax - onehot)/N`. Bad targets are a run-time error, `[E-SCE-TARGET]` |

### Comparisons

| Operation | Status | Notes |
|---|---|---|
| `greater`, `less`, `equal`, `greater_equal`, `less_equal`, `not_equal` | Supported | Output always f32 (0.0 / 1.0). Operands: equal shapes, or `(rows, cols)` against `(cols,)` |
| `>`, `<`, `==`, `>=`, `<=`, `!=` | Supported | Operator forms of the above, usable in conditions and as values: `let m = x >= y`. One comparison at a time; a numeric literal on either side is allowed |

---

## Control Flow

TensorLang supports `if` / `elif` / `else`. Conditions may be a tensor name, a boolean literal, or a comparison.

```text
let x: Tensor[f32, (1,)] = [1.0]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let c: Tensor[f32, (1,)] = [30.0]
let z: Tensor[f32, (1,)] = [0.0]

if x > 5.0 {
    let z = a
} elif x > 0.5 {
    let z = b
} else {
    let z = c
}
z
```

### Conditions

* Comparison operators: `>`, `<`, `==`, `>=`, `<=`, `!=`. The result is always f32. The same operators work as values (`let m = x >= y`) and as the named operations `greater_equal(x, y)` and so on.
* Either side of a comparison may be a tensor name or a numeric literal. A literal is lowered to an f32 fill with the same shape as the tensor on the other side, so `x > 0.5` compares every element of `x` with `0.5`. This works in `if` and `elif` conditions and in `let` comparisons. In a `let`, a literal can only be compared with a named tensor (not a nested call).
* A bare tensor name is also a valid condition.
* **Truthiness** is all-nonzero: a multi-element condition is true only if every element is nonzero.
* `-0.0` is falsy (it equals zero). An f16 value that underflows to zero when stored is zero, so it is falsy. Infinity is an ordinary value: `inf > 1.0` is true.
* **`NaN` is a run-time error** (`[E-NAN-IN-CONDITION]`). Every comparison with `NaN` is false (except `!=`), so a `NaN` loss would silently take the `else` branch. The operands of the condition are checked before it is evaluated, for comparisons, bare names, and `any()`/`all()`, in `if` and `elif`. This costs one device-to-host copy of each operand per evaluation, which matters only for large tensors in conditions.
* **An empty tensor is a compile error** (`[E-CONDITION-EMPTY]`). A zero-sized tensor has no elements to be "all nonzero".

### Branch consistency

A name defined in more than one branch must have the same shape and dtype in each. A mismatch is a compile error:

```text
if x > 0.0 {
    let y: Tensor[f32, (2, 2)] = add(x, x)
} else {
    let y: Tensor[f32, (1,)] = fill(1.0, (1,))   // [E-BRANCH-MISMATCH]
}
```

### Branches and loops

`if` statements may appear inside `for` loops; the condition is re-evaluated every iteration. Rebinds (`w = w_updated`) inside a selected branch follow the same pointer-swap rules as in loop bodies.

### Branches and autograd

`backward()` is supported in any branch of an `if`/`elif`/`else`, and in `if` statements nested inside branches. Only the nodes of the branch that actually ran (and of its enclosing branches) take part in the backward pass, so branches may define the same names with different operations:

```text
if c1 {
    let z = mult(w, w)
    let loss = sum(z)
    backward(loss)          // runs only if c1 is true
} elif c2 {
    let z = mult(w, x)
    let loss = sum(z)
    backward(loss)
} else {
    let z = add(w, w)
    let loss = sum(z)
    backward(loss)
}
```

* Each `backward()` runs at its position in the branch, in order, so `backward(loss)` followed by `sgd(w, w_grad, 0.01)` steps with the new gradient, and a branch may contain several `backward()` statements (the multi-pass rules above apply).
* The loss may depend on tensors computed before the `if`, or in an enclosing branch; their values are read from the device like at top level.
* A branch that is not taken never runs its backward pass, so the gradients of its parameters are not touched.

---

## Automatic Differentiation

TensorLang implements reverse-mode automatic differentiation. Tensors marked `with grad` accumulate gradients during `backward()`.

### Syntax

```text
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

let y = matmul(x, w)
let loss = sum(y)

backward(loss)
```

After `backward()`, gradients are available as synthesised variables:

```text
let w_grad = ...   // synthesised
let w_updated = minus(w, mult(lr, w_grad))
```

Gradients are saved as `name.grad.npy` and `name_grad.npy` when `--cache-layers` is used.

### Multiple backward passes

A file may contain more than one `backward()` statement. Each pass produces gradients for the parameters it depends on:

```text
let w1: Tensor[f32, (2, 2)] = fill(0.1, (2, 2)) with grad
let w2: Tensor[f32, (2, 2)] = fill(0.2, (2, 2)) with grad

let loss1 = sum(matmul(x, w1))
backward(loss1)
sgd(w1, w1_grad, 0.01)

let loss2 = sum(matmul(x, w2))
backward(loss2)
sgd(w2, w2_grad, 0.01)
```

A pass only overwrites the gradient (`<name>_grad`) of tensors its loss depends on. A parameter the loss does not reach keeps the gradient written by the last pass that did reach it, and a parameter that no pass has reached has a zero gradient. This means the updates can be applied after both passes:

```text
backward(loss1)
backward(loss2)

sgd(w1, w1_grad, 0.01)   // still the gradient from loss1
sgd(w2, w2_grad, 0.01)   // gradient from loss2
```

There is no implicit gradient zeroing. If you call `sgd(...)` or `adam(...)` on a parameter after a pass that did not reach it, its previous gradient is applied again, as in PyTorch without `zero_grad()`. Gradients written by a run do not carry over to the next compile.

Tested: separate parameters with an update after each pass, separate parameters with both updates deferred, and a parameter that no loss reaches (`autograd_multiple_backward`, `autograd_multiple_backward_deferred`, `autograd_unreached_param`).

### Supported Gradient Operations

| Operation | Gradient | Notes |
|---|---|---|
| `matmul(A, B)` | Yes | 2D and batched N-D operands |
| `add(A, B)` | Yes | Broadcast-aware reduction |
| `minus(A, B)` | Yes | Broadcast-aware reduction |
| `mult(A, B)` | Yes | Broadcast-aware reduction |
| `div(A, B)` | Yes | Broadcast-aware reduction |
| `relu` | Yes | |
| `sigmoid` | Yes | |
| `tanh` | Yes | |
| `softmax` | Yes | Any axis, N-D |
| `sum` | Yes | Any axis, N-D |
| `mean` | Yes | Any axis, N-D |
| `mse_loss` | Yes | N-D |
| `linear` | Yes | N-D (operates on last dim) |
| `layer_norm` | Yes | N-D, any axis |
| `batch_norm` | Yes | `dx`, `dgamma`, `dbeta` for the given-statistics forward |
| `instance_norm` | Yes | Simplified spatial-dim flattening |
| `cross_entropy` | Yes | Probability inputs, class-index targets |
| `softmax_cross_entropy` | Yes | Logit inputs, class-index targets; `(softmax - onehot)/N`, computed in float64 on the CPU |
| `gelu` | Yes | |
| `swish` | Yes | |
| `reshape` | Yes | N-D (gradient reshaped to input shape) |
| `transpose` | Yes | N-D (inverse permutation) |
| `slice` | Yes | N-D (gradient scattered into zeros) |
| `concat` | Yes | N-D (gradient split along concat axis) |
| `embedding` | Yes | Scatter-add into the weight (a repeated id accumulates); no gradient to the indices |
| `max` | Yes | Full and axis reduction; gradient flows to all extrema |
| `min` | Yes | Full and axis reduction; gradient flows to all extrema |
| `argmax` | Yes | Full and axis reduction; gradient is one-hot at index |
| `argmin` | Yes | Full and axis reduction; gradient is one-hot at index |
| `greater`, `less`, `equal`, `greater_equal`, `less_equal`, `not_equal` | Yes (zero) | Non-differentiable; gradient is zero. Other paths through the same tensor still receive gradient (`autograd_comparison_ops`) |
| `any`, `all` | Yes (zero) | Non-differentiable; gradient is zero |

Autograd works with both `f32` and `f16` tensors. For `f16`, gradient computation uses NumPy's type promotion (may upcast to f32 internally).

### Autograd Notes

* `matmul` gradients support 2D and batched N-D operands (broadcast dims handled via `_unbroadcast`).
* Broadcast-aware gradient reduction is implemented via `_unbroadcast()`.
* `cross_entropy` expects probability tensors, not logits.
* `cross_entropy` is not for training after `softmax`: its backward is `(p - onehot)/N`, the gradient with respect to *logits*, but it receives probabilities, so `softmax` -> `cross_entropy` back-propagates a wrong gradient (`tests/softmax_cross_entropy_chain.tl`). Train with `softmax_cross_entropy(logits, targets)` instead.
* `argmax`/`argmin` gradients use one-hot scatter (non-standard but useful for hard selection).
* `max`/`min` reduction gradients flow to all tied extrema (not divided).
* `instance_norm` gradients are simplified (see [Known limitations](#known-limitations)).

---

## Mixed Precision

FP16 training uses two features. Both are described in full, with the algorithm, Python API and limitations, in [docs/mixed_precision.md](docs/mixed_precision.md).

* **Loss scaling** scales the loss during `backward()` so small FP16 gradients do not underflow, then unscales before the optimizer step. It is off by default (scale `1.0`) and is set per file with a pragma:

```text
# loss_scale: 65536
# loss_scale: auto
# loss_scale: dynamic [init_scale] [patience]
```

  `auto` picks `65536.0` if any tensor in the graph is f16 and `1.0` otherwise. `dynamic` halves the scale and skips the optimizer step on overflow, and doubles it after `patience` clean steps (default 2000).
* **FP16 master weights** need no configuration. An f16 parameter declared `with grad` gets an FP32 master copy; `sgd` and `adam` update the master (with FP32 `m` and `v`) and cast the result back to f16 for the forward pass. FP32 parameters are unaffected.

Both work inside `for` loops. One gap: loss scaling with more than one `backward()` in a file is not yet tested.

---

## Training Loops

TensorLang supports multi-step gradient descent via `for` loops, including nested loops. For losses, optimizers, the opt-in GPU backward and how to verify a training program, see [docs/training.md](docs/training.md).

```text
for epoch in range(5) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
    save(w, "w.npy")
}
```

Nested loops are supported with independent counters:

```text
for i in range(3) {
    for j in range(4) {
        let inc = fill(1.0, (1, 1))
        let acc_new = add(acc, inc)
        acc = acc_new
    }
}
```

### Weight Update Styles

Flat bindings:

```text
let grad_step = mult(lr, w_grad)
let w_updated = minus(w, grad_step)
w = w_updated
```

Inline expression:

```text
let w_updated = minus(w, mult(lr, w_grad))
w = w_updated
```

Both compile to the same CUDA kernels. Inline expressions are flattened into synthetic `__tmp_N` bindings.

### Training Loop Features

* **Nested loops** — arbitrary depth, independent counters per level
* **`save()` inside loops** — fires per iteration, captures current storage, overwrites the file each time
* **`load()` inside loops** — data is loaded and uploaded to the GPU each iteration; the loaded shape is checked against the declared shape
* **Alias resolution** — `let y = x` inside a loop re-resolves after rebinds, so `y` tracks the rebound variable
* **Shape-stability guard** — rebinds with mismatched shapes are skipped with a diagnostic
* **Kernel symbols** generated inside loop bodies are scoped with a `loopN_` prefix, so the same tensor name can be reused across loop iterations, multiple loops, and top-level code without CUDA symbol collisions
* **Loss scaling** — works inside loops via `# loss_scale` pragma
* **FP16 master weights** — works inside loops; master buffer persists across iterations
* **Control flow** — `if` / `elif` / `else` may appear in loop bodies and is re-evaluated each iteration

### Training Loop Constraints

* Loop counts must be integer literals: `range(5)`, not `range(n)`.
* Rebinding requires a name on the right-hand side: `w = w_updated`, not `w = minus(w, step)`.
* Tensors declared inside a loop body are not persisted to the `.npy` cache; recompute them after the loop under a distinct name to inspect them.

### Operations Supported in Loop Bodies

Loop bodies have operation parity with top level, including:

* `add`, `minus`, `mult`, `div`
* `matmul`
* `relu`, `sigmoid`, `tanh`, `gelu`, `swish`
* `softmax`
* `mse_loss`, `cross_entropy`
* `linear`
* `sum`, `mean`, `min`, `max`, `argmin`, `argmax`
* `layer_norm`, `batch_norm`, `instance_norm`
* `greater`, `less`, `equal`
* slicing, `reshape`, `transpose`, `concat`
* `fill`, tensor literals
* `load()` and `save()`
* user-defined functions
* nested `for` loops
* `if` / `elif` / `else`
* `let` bindings and rebinds
* `backward()`

### Rebinding

```text
w = w_updated
```

This swaps GPU buffer pointers rather than copying memory.

---

## Optimizers

TensorLang provides built-in optimizers (`sgd`, `adam`) with compiler-managed state. An optimizer is a statement in the loop body, placed after `backward()`; it updates the parameter in place from its gradient `<name>_grad`.

### Syntax

Arguments are positional: `sgd(param, param_grad, lr)` and `adam(param, param_grad, lr, beta1, beta2, eps)`, where `beta1`, `beta2` and `eps` default to 0.9, 0.999 and 1e-8.

```text
let w: Tensor[f32, (1, 1)] = [[0.0]] with grad

for epoch in range(100) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    adam(w, w_grad, 0.01)
}
```

SGD is used the same way:

```text
    sgd(w, w_grad, 0.01)
```

### Features

* **Compiler-managed state** — Adam allocates its `m` and `v` buffers once per parameter and keeps them across iterations
* **FP32 master weights** — for FP16 params, the optimizer computes in FP32 with FP32 `m`/`v` buffers and casts to FP16 for the forward pass
* **Bias correction** — computed on the host in float64 (v0.5.9; earlier versions computed `1 - beta^t` in float32 on the device, which was off by up to 6e-5 for `beta2 = 0.999`)

### Optimizer Names

The following built-in optimizer constructs are recognised by the compiler:

* `sgd`
* `adam`
* `sgd_update`
* `adam_update`

---

## Data-Parallel Training

```bash
python3 tensorlang.py --ddp --nproc 2 train.tl                    # one worker per GPU
python3 tensorlang.py --ddp --nproc 4 --ddp-same-gpu train.tl     # several workers on one GPU (development)
```

`--ddp` starts N copies of the same command, each pinned to its own GPU with `RANK` and `WORLD_SIZE` set. Each worker trains on its own shard of the data. After every `backward()` the workers average the gradients of their `with grad` tensors and agree on the overflow flag, so all of them take the same optimizer step; at start-up they all take rank 0's parameters. The program chooses the shard: `{rank}` and `{world_size}` in `load()` and `save()` paths are replaced for each worker (by `0` and `1` in an ordinary run).

```text
let x: Tensor[f32, (32, 8)] = load("data/x_{rank}.npy")
let w: Tensor[f32, (8, 1)] = load("w.npy") with grad
...
save(w, "w_final.npy")      # lands in cache/<program>/ws2/rank0/ and .../rank1/
```

The averaged gradient equals the whole-batch gradient when the shards are the same size and the loss is a mean (`mse_loss`, `softmax_cross_entropy`, `mean`). Gradients are exchanged through host memory over TCP on one machine; there is no NCCL yet. `tools/ddp_check.py` trains a small network on the whole batch and on shards: the workers end bit-identical and match single-process training to about 5e-8 (SGD, a two-layer GELU network, two GTX 1080 Ti). Checks on a transformer with Adam, and a speed measurement that includes the exchange, are still to do. The launcher, the environment variables, the cache layout and the error messages are in [docs/data_parallel.md](docs/data_parallel.md).

---

## File I/O

### Load

```text
let data: Tensor[f32, (10, 4)] = load("data.npy")
```

Supported formats:

* `.npy`
* `.npz` (uses `data` array or first available array)
* `.csv` (requires pandas; drops a leading date/time/index/id column)

If no type annotation is provided, shape and dtype are inferred from the file: a float16 file loads as `f16`; every other numeric dtype, including NumPy's default float64, loads as `f32` (converted on load, with a note under `--debug-info`).

`load()` works at top level and inside `for` loop bodies. Paths are relative to the working directory the compiler is run from. `{rank}` and `{world_size}` in the path are replaced by the worker's values (see [Data-Parallel Training](#data-parallel-training)).

### Save

```text
save(tensor, "output.npy")
```

`save()` is supported both at top level and inside `for` loops. Inside a loop, it fires per iteration and overwrites the file each time, so the final file reflects the last iteration's value.

---

## NumPy Interop

TensorLang provides zero-copy NumPy interop at the Python API level. CPU-side tensor buffers can be shared with NumPy arrays without copying.

### Python API

```python
import numpy as np
from tensorlang.compiler import TensorCompiler

compiler = TensorCompiler()

# Zero-copy load: TensorLang uses arr's buffer directly
arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
compiler.load_numpy("x", arr)

# Zero-copy view: returns np.ndarray sharing storage with tensors["x"]
view = compiler.to_numpy("x")
assert view.ctypes.data == arr.ctypes.data

# Explicit sync helpers
compiler.sync_to_gpu("x")   # push CPU buffer → GPU
compiler.sync_to_cpu("x")   # pull GPU → CPU (in place)
```

### Contracts

| Direction | Ownership | Lifetime | Mutation visibility |
|---|---|---|---|
| NumPy → TensorLang | NumPy owns CPU buffer; TensorLang owns GPU copy | NumPy array must outlive the compiler session | Python writes visible only if re-synced to GPU |
| TensorLang → NumPy (CPU) | TensorLang owns `tensors[name]`; NumPy gets a view | Compiler session must outlive the NumPy view | TensorLang writes (via kernels) visible through the view after D→H sync |

### Constraints

* `load_numpy` accepts `float16`, `float32`, and `float64` arrays; any other dtype raises `TypeError`. A non-contiguous array is first copied to a contiguous one, so that array is no longer shared with NumPy (`to_numpy` returns the copy).
* GPU-side zero-copy (CUDA unified memory) is deferred to a future version.
* The interop helpers move buffers between NumPy and the GPU. They do not feed `.tl` programs: `compile_and_execute` starts from a fresh environment, so a `load_numpy` buffer is not visible to a program. Buffers of all three dtypes can be shared and synced, but `f64` has no kernel support (see [Current Status and Limitations](#current-status-and-limitations)).

---

## Cache

TensorLang stores compilation artifacts under:

```text
cache/<input-file>/
```

Contents include:

* `kernel.cu` — generated CUDA source
* `kernel.so` — compiled shared library
* `tensor_name.npy` — computed tensor results
* `tensor_name.grad.npy` — gradients, when `--cache-layers` is used
* `tensor_name_grad.npy` — synthesised gradient variables, when `--cache-layers` is used

The full test suite (`bash build.sh --test`) clears `cache/` before running. In addition, the test runner removes each test's own cache directory immediately before running it, so a filtered run (`--test --filter ...`) cannot pass on leftover artifacts from an earlier run.

One exception to be aware of: `load_basic`, `load_dynamic`, and `load_inferred` read their inputs from `cache/tests/basics.tl/`, which is another test's output. Run the full suite once (or run `basics.tl` first) before running those three on their own.

For an isolated run outside the test runner, clear the entry manually:

```bash
rm -rf cache/tests/my_test.tl/
python3 tensorlang.py tests/my_test.tl --cache-layers
```

---

## Diagnostics

Errors carry a stable code. Type-checker errors print as `[TYPE CHECKER] error: [E-CODE] message` and parse errors as `[COMPILER] error: [E-CODE] message`. The code is stable; the wording may be improved between releases, so scripts and tests should match on the code.

| Code | Raised for |
|---|---|
| `[E-SHAPE-MISMATCH]` | Shape errors in elementwise ops, `matmul`, comparisons, and other shape-checked operations |
| `[E-BRANCH-MISMATCH]` | A name defined in more than one `if`/`elif`/`else` branch with different shapes or dtypes |
| `[E-UNDEFINED-TENSOR]` | A tensor name used before it is defined (operation args, conditions) |
| `[E-CONDITION-OPERAND]` | A condition comparison whose operand is neither a tensor name nor a numeric literal |
| `[E-SCALAR-LITERAL]` | A bare number bound with `let`, or a numeric literal used as an operand |
| `[E-CONDITION-EMPTY]` | An empty (zero-sized) tensor used in an `if`/`elif` condition |
| `[E-NAN-IN-CONDITION]` | A `NaN` operand in an `if`/`elif` condition. A run-time error, printed as `[RUNTIME] error: ...` |
| `[E-EMBEDDING-SHAPE]` | An `embedding` weight that is not 2-D, indices with no dimensions, or the wrong number of arguments |
| `[E-EMBEDDING-DTYPE]` | `embedding` indices that are not `f32` |
| `[E-EMBEDDING-INDEX]` | An `embedding` index that is not an integer in `[0, vocab)`. A run-time error, printed as `[RUNTIME] error: ...` |
| `[E-SYNTAX]` | Any other syntax error, including chained comparisons (`a > b > c`); reported with the source line and a caret where there is one |

Messages name the expression involved and avoid compiler internals. A nested call such as `add(x, mult(x, nope))` is reported as ``the nested expression `mult(x, nope)` ``, not as a `__tmp_N` temporary, and a near-miss name gets a suggestion:

```text
[TYPE CHECKER] error: [E-UNDEFINED-TENSOR] `let y = add(weights, wieghts)` uses 'wieghts', which is not defined at this point. Did you mean 'weights' instead of 'wieghts'?

[TYPE CHECKER] error: [E-BRANCH-MISMATCH] branch shape/dtype mismatch for 'y': the `if` branch defines it as f32 (2, 2), but the `else` branch defines it as f32 (1,). A name used after an if/elif/else must have the same shape and dtype in every branch that sets it.

[TYPE CHECKER] error: [E-SHAPE-MISMATCH] matmul: shape mismatch computing 'z': 'x' (3, 4) vs 'y' (5, 6). Inner dimensions must match: left has 4 columns but right has 5 rows

[COMPILER] error: [E-SCALAR-LITERAL] a numeric literal cannot be used here (line 2, column 16).
let y = add(x, 2.0)
               ^
Numeric literals are only valid in comparison conditions (`if x > 0.5`) and in optimizer arguments (`sgd(w, w_grad, 0.01)`). To use a constant as an operand, declare a tensor, e.g. `let half: Tensor[f32, (1,)] = [0.5]`.
```

A failed compile exits with a non-zero status. A Python traceback is printed for parse errors only with `--debug`. Each code has a negative test under `tests/diagnostics_*.tl`.

---

## Architecture

### Compilation Pipeline

1. **Lexing & Parsing** — Lark grammar (`tensorlang.lark`)
2. **AST Construction** — flatten inline expressions; inline user-defined functions; build nested loop bodies
3. **Type Checking** — validate tensor shapes, broadcasting rules, branch consistency, gradient tensor pre-registration, dtype propagation and promotion (N-D aware)
4. **CUDA Generation** — emit forward kernels; autograd is handled by the NumPy reference graph plus compiler-recorded operations; elementwise fusion (including broadcasting and N-D); dtype-aware kernel generation (f16/f32); stride-based N-D indexing; loss scaling (pragma-parsed, scale passed to backward, unscale before optimizer); FP16 master weights (master buffer allocation, init, mixed-type optimizer kernels)
5. **Compilation** — `nvcc` to shared library
6. **Execution** — PyCUDA kernel launching, memory management, loop iteration with pointer-swap rebinding, nested loop recursion, `save()` per iteration, branch selection

### Key Modules

* `tensorlang/ast_builder.py` — AST construction and inlining
* `tensorlang/type_checker.py` — shape and broadcasting checks (N-D aware), branch consistency, diagnostic codes
* `tensorlang/kernel_generator.py` — CUDA kernel generation
* `tensorlang/codegen.py` — code generation orchestration
* `tensorlang/compiler.py` — orchestration, execution, caching, scoped kernel names, loss scale resolution, master weight allocation
* `tensorlang/execution.py` — runtime execution, loops, branches, rebinding
* `tensorlang/interop.py` — NumPy zero-copy interop (`InteropMixin`)
* `tensorlang/autograd.py` — NumPy reference autograd implementation (supports `scale` param, overflow detection)
* `tensorlang/fusion.py` — elementwise kernel fusion (broadcasting-aware, N-D aware)
* `tensorlang/generators/` — per-category CUDA kernel generators (base, comparisons, elementwise, nn, optimizers, reductions, shapes, unary) — all dtype-aware and N-D aware
* `tensorlang/test_runner.py` — test discovery and execution
* `tensorlang/app_runner.py` — application discovery and execution
* `tensorlang/gpu.py` — GPU detection and process-level pinning
* `tensorlang/gpu_backward.py` — opt-in GPU backward for `matmul` and `gelu`, with a device buffer pool
* `tensorlang/dist.py` — data-parallel collectives over host TCP (`all_reduce`, `broadcast`, `all_gather`, `barrier`), gradient averaging, `{rank}` placeholders
* `tensorlang/launcher.py` — the `--ddp` launcher (one worker per GPU, failure and signal handling)
* `tensorlang/kernel_cache.py` — shared kernel build and per-rank cache directories for data-parallel runs
* `tensorlang/tensor_lang.py` — argument parsing
* `tensorlang/tensor_verifier.py` — tensor verification helpers
* `tensorlang/constants.py` — shared constants (dtype registry, promotion, loss scale defaults)
* `tensorlang/io.py` — file I/O helpers

---

## Performance

Current performance characteristics:

* Matrix multiplication is backed by cuBLAS (replacing the earlier shared-memory tiled kernel), including batched N-D via strided batched GEMM.
* Elementwise chains such as `add → relu` are fused into a single CUDA kernel, including broadcasting cases (row/column/general broadcast modes) and N-D.
* Training-loop weight updates use pointer swaps (no GPU memory copy between iterations).
* Generated CUDA source is cached and reused when unchanged.
* Mixed precision: FP16 storage with FP32 accumulation reduces memory bandwidth; cuBLAS tensor-core matmul path for f16 (engages on Volta+; falls back to standard FP16 GEMM on Pascal).
* N-D operations use stride-based indexing with per-element thread dispatch.
* GPU backward (opt-in `TENSORLANG_GPU_BACKWARD=1`): `matmul` and `gelu` backward on the GPU with a device buffer pool. The 20-step Adam run of the 4-layer micro-GPT of `docs/training.md` takes about 1.8 s, against about 4.7 s with the NumPy backward (GTX 1080 Ti).
* Data-parallel (`--ddp`): two workers on two GTX 1080 Ti each ran that program at about single-process speed (1.84 s and 2.03 s, measured before the gradient exchange was added), so two cards do about 1.7 to 1.8 times the work of one. The cost of the gradient exchange at scale has not been measured. The launcher gives each worker its share of the CPU's BLAS threads (two NumPy-backward workers took 15.8 s each without it and 4.5 s with it).
* 4096×4096 matrix multiplication runs, but has not been benchmarked against cuBLAS or PyTorch.

Performance claims should be treated as unverified until baselined on the same hardware.

---

## Testing

### Single File

```bash
python3 tensorlang.py tests/activation_pipeline.tl
python3 tensorlang.py tests/inline_expr_basic.tl --debug
```

### Full Test Suite

```bash
bash build.sh --test
source build.sh --test --filter autograd
source build.sh --test --filter for_loop
source build.sh --test --filter control_flow
```

`--filter` matches on the test name, so tests are named by feature: `control_flow_*`, `for_loop_*`, `autograd_*`, `load_*`, `any_all_*`, and so on. Do not prefix tests with a version number.

A GPU-free helper, `python3 check.py path/to/file.tl`, parses and type-checks a file without running it.

Test annotations (`@EXPECTED`, `@EXPECT_FAILURE`), tensor verification, the test inventory, the helper tools and multi-GPU test execution are in [docs/testing.md](docs/testing.md). Running real GPT-2 is in [docs/gpt2_demo.md](docs/gpt2_demo.md), and editing its weights is in [docs/weight_space_programming.md](docs/weight_space_programming.md).

---

## Applications

Applications live under `apps/` and are discovered by `app.toml`. The shared toolkit `apps/tlkit/` (`chunked_runner.py`, `sequence_player.py`, `colormap.py`) supports visual/interactive apps.

```bash
python3 tensorlang.py --list-apps
python3 tensorlang.py --app examples/hello_mlp
python3 tensorlang.py --app examples/decision_boundary
python3 tensorlang.py --app games/2048 --app-args <16 numbers>
```

Layout, the `app.toml` format, requirements and app tests are in [docs/apps.md](docs/apps.md).

### Notable Apps

* `examples/hello_mlp` — minimal 2-layer MLP training example.
* `examples/adaptive_lr` — two linear models in one loop, each with its own loss, `backward()` and `sgd` step; `if loss1 > 1.0` switches model 1 between a large and a small learning rate, taken in some iterations and not others.
* `examples/decision_boundary` — trains a 2-8-1 tanh/sigmoid MLP on XOR and animates the decision boundary. Verified: loss dropped from 0.243585 to 0.001573 over 1000 epochs.
* `games/tic_tac_toe` — complete.
* `games/2048` — tensor-ops game engine (`step.tl`) verified.

---

## Future Work

Shipped releases are recorded in [RELEASES.md](RELEASES.md). This section is for work that has not shipped.

### v0.5.11 (planned) — training that stays on the GPU, and the smaller gaps

* Stage 2 of the GPU backward: keep gradients and activations on the device between ops (today each `matmul` backward uploads three arrays and downloads two, and every kernel result is copied to the host). Measured on the micro-GPT (`docs/training.md`): copies are about a quarter of a step and the GPU wait about 1%, so making the synchronisation optional is not worth doing. Once gradients stay on the device, data-parallel runs can move from host TCP to NCCL
* Data-parallel follow-ups: `tools/ddp_check.py` on a transformer (Adam, `layer_norm`, `softmax`, `embedding`); the speed-up with the gradient exchange on `micro_gpt_perf`; explicit collectives in the language (`all_reduce`, `all_gather`)
* A multi-head training generator (fused QKV, slicing, like `probe_gpt2_multihead`), and `load()` indexed by the loop counter, as the two prerequisites for fine-tuning GPT-2 small
* Elementwise `sqrt`, `rsqrt`, `exp`, `log` (needed for RMSNorm and so for LLaMA-style models), with gradients

* In-language generation: keep the weights resident and run the whole loop in one process (a fixed-length padded token tensor, row selection and update by one-hot arithmetic, no growing rebind)
* `where(cond, a, b)`: elementwise select, for clean masking
* Allow `fill` (and other special forms) as an operand inside another call
* Proposed: a coded diagnostic for a repeated top-level `let` of the same name (loop bodies, and the same name in sibling branches, stay legal), after a sweep of the existing tests and apps
* `load()` selecting an array by name in an `.npz`; a built-in tanh option for `gelu`
* Document: distinct names versus rebinds

**Phase D — only with evidence (deferred)**
* A fused attention kernel, if profiling shows the composed version is the bottleneck
* Quantised inference (needs a packed or ternary dtype and a checkpoint loader)
* CUDA backward for the remaining ops, if training throughput is the goal (`matmul` and `gelu` can run on the GPU since v0.5.9; the rest is NumPy on the CPU)
* `arange` / `linspace` for creating varied tensors

### Short Term

* Improve shape-mismatch error messages
* Dynamic loss scaling tuning (growth interval, overflow threshold)

### Medium Term

* GPU-side zero-copy interop (CUDA unified memory)
* Larger NumPy interop round-trip tests
* mmap-based zero-copy `load()` in `.tl`
* Hot reload in dev mode
* More practical example applications (vision, simple game agent, signal processing, finance) so the language does not feel purely abstract

### Long Term

* Multi-GPU single-program execution (Tier-B: one model spanning GPUs; data-parallel `--ddp` exists since v0.5.10)
* MLIR integration
* Additional hardware backends (ROCm, Metal)
* WGSL transpile backend (or removal of the `--transpile` flag)

---

## Contributing

Contributions are welcome. Useful areas include:

* New operations and their gradients
* Kernel optimisations
* Tests with exact `@EXPECTED` values from real runs
* Documentation improvements
* Application examples

When adding tests:

* Copy expected values from a real run. Do not manually round or approximate them.
* Name tests by feature (`control_flow_*`, `for_loop_*`, `autograd_*`), not by version.
* For negative tests, pin a diagnostic code: `// @EXPECT_FAILURE: [E-...]`.
* Run the test on the GPU, not only through `check.py`; some bugs only show at runtime.

---

## About

Original author: davro (David). Related projects of interest: `workspace` (PyQt6 IDE with Ollama), `we-not-me` (TradingView indicators). Primary motivation for recent work: move from pure compiler infrastructure toward practical, visible applications.

---

## License

Licensed under the GNU Lesser General Public License v3 (LGPL-3.0). See `LICENSE` for details.

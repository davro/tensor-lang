# TensorLang: A Native AI/ML Language

TensorLang is an open-source, native machine-learning language designed to compile tensor operations into GPU-accelerated CUDA kernels.

This project addresses pain points in Python-based ML frameworks by offering a streamlined, hardware-aware alternative for high-performance tensor computations. The compiler is implemented in Python, but the hot path — tensor operations — runs as compiled CUDA kernels.

> **Status:** v0.5.3 in progress (for-loop body parity complete; control-flow syntax/AST staged). Core suite: 170/170 `.tl` tests + 31/31 Python tests passing.
> **Hardware validated on:** 2× NVIDIA GTX 1080 Ti (Pascal, sm_61).
> **Current focus:** v0.5.3 — control-flow execution and remaining parity items.

---

## Table of Contents

* [Overview](#overview)
* [Current Status and Limitations](#current-status-and-limitations)
* [Setup](#setup)
* [Quick Start](#quick-start)
* [Repository Layout](#repository-layout)
* [Language Overview](#language-overview)
* [Operations](#operations)
* [Automatic Differentiation](#automatic-differentiation)
* [Loss Scaling](#loss-scaling)
* [Training Loops](#training-loops)
* [Optimizers](#optimizers)
* [File I/O](#file-io)
* [NumPy Interop](#numpy-interop)
* [Cache](#cache)
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
* **ML-first design** — built specifically for neural network operations
* **Automatic differentiation** — reverse-mode autograd for a broad set of operations (FP32 and FP16)
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
* Comparisons: `greater`, `less`, `equal` (N-D broadcasting)
* Slicing, reshape, transpose, concat (N-D aware)
* Normalisations: `layer_norm`, `batch_norm`, `instance_norm` (N-D aware)
* Losses: `mse_loss`, `cross_entropy` (N-D aware)
* Linear layers: `linear` (N-D aware)
* Reverse-mode autograd for a broad set of operations (FP32 and FP16)
* Training loops with weight rebinding, nested loops, `save()` inside loops, and alias resolution
* Built-in optimizers: `sgd`, `adam` (compiler-managed state, multi-parameter, FP32 master weights for FP16 params)
* Elementwise kernel fusion for elementwise chains, including broadcasting (row/column/general broadcast modes, N-D aware)
* Scoped kernel names for loop bodies
* User-defined functions
* File I/O: `load()`, `save()`
* NumPy zero-copy interop (CPU-side buffer sharing, float16/32/64)
* Comments: `//` and `/* */`
* Application system (`apps/`) with `hello_mlp`, `decision_boundary`, `tic_tac_toe`, and `2048` (engine verified)
* Shared app toolkit (`apps/tlkit/`)
* **Mixed precision (FP16/FP32)** — FP16 storage with FP32 accumulation; cast-based kernels; cuBLAS tensor-core matmul path; f16 autograd and training loops
* **N-dimensional tensor support** — general stride-based indexing across elementwise, reductions, shape ops, normalisations, activations, and batched matmul
* **Autograd coverage expansion** — gradients now wired for `reshape`, `transpose`, `slice`, `concat`, `max`, `min`, `argmax`, `argmin`, and comparisons (zero gradient for non-differentiable ops)
* **Loss scaling** — static and dynamic loss scaling for FP16 training stability; configurable via `# loss_scale: <value|auto|dynamic>` pragma; gradients upcast to FP32 when scale overflows FP16 max; overflow detection with automatic scale reduction (halve on overflow, double after N successful steps); unscale before optimizer step
* **FP16 master weights** — Adam/SGD with FP32 master weights for FP16 params; optimizer computes in FP32 with FP32 m/v buffers, casts to FP16 for forward pass; automatic for FP16 params with grad
* **Batched matmul gradients** — N-D (batched) matmul backward via NumPy batched matmul + `_unbroadcast` for broadcast dims
* **For-loop body parity** — normalisations, comparisons, slicing, reshape, transpose, concat, and additional reductions now execute in loop bodies

### Known limitations

* **Control flow:** `if` / `elif` / `else` is present in the grammar and AST construction, but is **not currently executed**. Targeted for v0.5.3.
* **`any()` / `all()` reductions:** present in the grammar but **not implemented**. Targeted post-v0.5.
* **`f64`:** parsed by the grammar, but type checking and kernels effectively force `f32`. Intentional for an ML language; full f64 support is not planned.
* **For-loop bodies:** broad operation parity with top-level is now implemented; remaining gaps are user-defined functions and file I/O other than `save()`.
* **Axis support is limited** for several normalisations.
* **`batch_norm`** uses running mean/variance; it does **not** apply `gamma`/`beta` affine parameters. N-D uses simplified NCHW convention.
* **`instance_norm`** N-D uses simplified spatial-dim flattening.
* **Dev mode hot reload** is not implemented.
* **`--transpile` (WGSL)** is exposed but not currently functional.
* **NumPy interop is CPU-side only** — GPU-side zero-copy requires CUDA unified memory (deferred).
* **Single backward pass per compilation** — multiple `backward()` statements in one file are not supported.
* **Autograd N-D limitations** — `batch_norm`/`instance_norm` gradients are simplified.
* **Dynamic loss scaling** — overflow detection works but scale growth interval defaults to 2000 steps; may need tuning for short training runs.

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
│   ├── interop.py             # NumPy zero-copy interop
│   ├── io.py                  # File I/O helpers
│   ├── kernel_generator.py    # CUDA kernel generation
│   ├── tensor_lang.py         # Argument parsing etc.
│   ├── tensor_verifier.py     # Tensor verification helpers
│   ├── test_runner.py         # Test discovery and execution
│   └── type_checker.py        # Shape and broadcasting checks
├── tensorlang.lark            # Grammar
├── build.sh                   # Install / test / lint helper
├── tests/                     # Core language tests (.tl)
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

Supported dtypes: `f16`, `f32`. (`f64` is parsed but lowered to `f32`.)

Shapes are part of the type system. The compiler checks shape compatibility and broadcasting rules at compile time, including N-D broadcasting (NumPy-style: align trailing dims, broadcast size-1 dims).

### User-Defined Functions

```text
fn normalize(x: Tensor[M, N]) -> Tensor[M, N] {
    let mu = mean(x, axis=1)
    return minus(x, mu)
}
```

Function calls are inlined. Generic dimensions such as `M` and `N` are primarily syntactic; concrete shapes are resolved at call sites.

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
| `batch_norm` | Partial | N-D (channel at dim 1, NCHW); uses running mean/variance; no gamma/beta |
| `instance_norm` | Partial | N-D (per-instance over non-batch dims) |

### Losses

| Operation | Status | Notes |
|---|---|---|
| `mse_loss` | Supported | N-D; predictions and targets must have the same shape. Result is a `(1,1)` tensor, not a bare scalar — use `.item()` in Python. |
| `cross_entropy` | Supported | N-D; probability inputs and class-index targets (last dim = classes) |

### Comparisons

| Operation | Status | Notes |
|---|---|---|
| `greater`, `less`, `equal` | Partial | N-D broadcasting; output always f32 |

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
| `cross_entropy` | Yes | Probability inputs, class-index targets |
| `gelu` | Yes | |
| `swish` | Yes | |
| `reshape` | Yes | N-D (gradient reshaped to input shape) |
| `transpose` | Yes | N-D (inverse permutation) |
| `slice` | Yes | N-D (gradient scattered into zeros) |
| `concat` | Yes | N-D (gradient split along concat axis) |
| `max` | Yes | Full and axis reduction; gradient flows to all extrema |
| `min` | Yes | Full and axis reduction; gradient flows to all extrema |
| `argmax` | Yes | Full and axis reduction; gradient is one-hot at index |
| `argmin` | Yes | Full and axis reduction; gradient is one-hot at index |
| `greater` | Yes (zero) | Non-differentiable; gradient is zero |
| `less` | Yes (zero) | Non-differentiable; gradient is zero |
| `equal` | Yes (zero) | Non-differentiable; gradient is zero |

Autograd works with both `f32` and `f16` tensors. For `f16`, gradient computation uses NumPy's type promotion (may upcast to f32 internally).

### Not Currently Wired into Autograd

The following operations are present in the language but are **not currently recorded for gradient computation**:

* `batch_norm`
* `instance_norm`

### Autograd Notes

* `matmul` gradients support 2D and batched N-D operands (broadcast dims handled via `_unbroadcast`).
* Broadcast-aware gradient reduction is implemented via `_unbroadcast()`.
* `cross_entropy` expects probability tensors, not logits.
* `argmax`/`argmin` gradients use one-hot scatter (non-standard but useful for hard selection).
* `max`/`min` reduction gradients flow to all tied extrema (not divided).

---

## Loss Scaling

TensorLang supports **static and dynamic loss scaling** for mixed-precision (FP16) training stability. Loss scaling prevents gradient underflow by scaling the loss (and thus all gradients) by a large factor during backward, then unscaling before the optimizer step.

### How It Works

1. During `backward()`, the loss gradient is seeded with `scale` instead of `1.0`
2. All gradients are scaled by `scale` during backpropagation
3. Before the optimizer step, gradients are divided by `scale` (unscaled)
4. If the scale exceeds FP16 max (~65504), gradients are automatically upcast to FP32
5. **Dynamic mode:** overflow (inf/nan in gradients) is detected after backward; scale is halved and optimizer step is skipped; after N successful steps, scale is doubled

### Configuration

**Via pragma (in `.tl` file):**

```text
# loss_scale: 65536
let x: Tensor[f16, (4,)] = [1.0, 2.0, 3.0, 4.0] with grad
let y = mult(x, x)
let loss = sum(y)
backward(loss)
```

**Auto-detect:**

```text
# loss_scale: auto
```

Auto-detect checks if any tensor in the graph is FP16 and selects `65536.0`; otherwise uses `1.0`.

**Dynamic scaling:**

```text
# loss_scale: dynamic [init_scale] [patience]
```

Enables dynamic scaling with optional initial scale (default: auto-detected) and patience (default: 2000 successful steps before doubling).

**Via Python API:**

```python
compiler = TensorCompiler(loss_scale=65536.0)
compiler = TensorCompiler(dynamic_loss_scale=True, loss_scale_patience=100)
compiler.compile_and_execute("model.tl")
```

**Default:** `1.0` (no scaling). Existing tests and FP32 programs are unaffected.

### Supported Scales

| Dtype | Default Scale | Notes |
|---|---|---|
| `f32` | `1.0` | No scaling needed |
| `f16` | `65536.0` (2^16) | Auto-detected or explicit; gradients upcast to FP32 |

### Loss Scaling in Training Loops

Loss scaling works inside `for` loops. The scale is applied per-backward-call and unscaled before the optimizer step:

```text
# loss_scale: 65536
let w: Tensor[f16, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

for epoch in range(10) {
    let y = matmul(x, w)
    let loss = mse_loss(y, y_true)
    backward(loss)
    opt.step()
}
```

### Limitations

* **Single backward pass** — multiple `backward()` statements in one file are not supported.
* **FP16 master weights** — ✅ now supported (see below).

---

## FP16 Master Weights

TensorLang supports **FP32 master weights** for FP16 parameters. When an FP16 tensor has `with grad`, the compiler automatically allocates an FP32 master weight buffer. The optimizer (Adam/SGD) computes updates in FP32 (m, v buffers in FP32) and casts the result back to FP16 for the forward pass.

### How It Works

1. When an FP16 tensor is declared with `with grad`, an FP32 master weight buffer (`{name}_master`) is allocated
2. Master is initialized from the FP16 param (cast FP16 → FP32)
3. During `opt.step()`, the optimizer updates the master weight (FP32) with FP32 m/v buffers
4. After update, master is cast to FP16 and written to the param buffer
5. The forward pass uses the FP16 param as before

### Usage

No configuration needed — master weights are automatic for FP16 params with grad:

```text
let w: Tensor[f16, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad
let opt = adam([w], lr=0.01)

for epoch in range(100) {
    let y = matmul(x, w)
    let loss = mse_loss(y, y_true)
    backward(loss)
    opt.step()  // updates master (FP32), casts to w (FP16)
}
```

FP32 params continue to use the original (non-master) code path. No extra memory or overhead for FP32 training.

### Benefits

* **Better precision** — optimizer state (m, v) computed in FP32 instead of FP16
* **No underflow** — weight updates in FP32 prevent FP16 underflow in small gradients
* **Transparent** — no API changes; automatic for FP16 params with grad
* **Compatible with loss scaling** — works with both static and dynamic loss scaling

---

## Training Loops

TensorLang supports multi-step gradient descent via `for` loops, including nested loops.

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
* **Alias resolution** — `let y = x` inside a loop re-resolves after rebinds, so `y` tracks the rebound variable
* **Shape-stability guard** — rebinds with mismatched shapes are skipped with a diagnostic
* **Kernel symbols** generated inside loop bodies are scoped with a `loopN_` prefix, so the same tensor name can be reused across loop iterations, multiple loops, and top-level code without CUDA symbol collisions
* **Loss scaling** — works inside loops via `# loss_scale` pragma
* **FP16 master weights** — works inside loops; master buffer persists across iterations

### Training Loop Constraints

* Loop counts must be integer literals: `range(5)`, not `range(n)`.
* Rebinding requires a name on the right-hand side: `w = w_updated`, not `w = minus(w, step)`.

### Operations Supported in Loop Bodies

Loop bodies now have broad operation parity with top-level, including:

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
* `fill`
* `save()`
* nested `for` loops
* `let` bindings and rebinds
* `backward()`

Remaining loop-body gaps are user-defined functions and file I/O other than `save()`.

### Rebinding

```text
w = w_updated
```

This swaps GPU buffer pointers rather than copying memory.

---

## Optimizers

TensorLang provides first-class built-in optimizers (`sgd`, `adam`) with compiler-managed state. Instead of hand-writing `w = w - lr * grad` for every parameter, declare an optimizer over a list of parameters and call `opt.step()`.

### Syntax

```text
let opt = adam([w1, w2], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)

for epoch in range(100) {
    let y_pred = matmul(x, w1)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    opt.step()
}
```

SGD is used the same way:

```text
let opt = sgd([w], lr=0.01)
```

### Features

* **Compiler-managed state** — Adam allocates and persists `m` and `v` buffers across iterations
* **Multi-parameter** — a single optimizer can update several parameters at once
* **Reuses pointer-swap rebinding** — parameter updates swap GPU buffer pointers (no copy)
* **Gradient zeroing / accumulation semantics** supported
* **FP32 master weights** — for FP16 params, optimizer computes in FP32 with FP32 m/v buffers, casts to FP16 for forward pass

### Optimizer Names

The following built-in optimizer constructs are recognised by the compiler:

* `sgd`
* `adam`
* `sgd_update`
* `adam_update`

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

If no type annotation is provided, shape and dtype are inferred from the file (including float16 from `.npy`).

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

* `load_numpy` requires C-contiguous `float16`, `float32`, or `float64` arrays (non-contiguous arrays are made contiguous automatically).
* GPU-side zero-copy (CUDA unified memory) is deferred to a future version.
* Mixed precision through shared buffers is supported (f16/f32/f64).

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

The full test suite clears `cache/` before running. For isolated runs after modifying a test, clear its cache entry manually:

```bash
rm -rf cache/tests/my_test.tl/
python3 tensorlang.py tests/my_test.tl --cache-layers
```

---

## Architecture

### Compilation Pipeline

1. **Lexing & Parsing** — Lark grammar (`tensorlang.lark`)
2. **AST Construction** — flatten inline expressions; inline user-defined functions; build nested loop bodies
3. **Type Checking** — validate tensor shapes, broadcasting rules, gradient tensor pre-registration, dtype propagation and promotion (N-D aware)
4. **CUDA Generation** — emit forward kernels; autograd is handled by the NumPy reference graph plus compiler-recorded operations; elementwise fusion (including broadcasting and N-D); dtype-aware kernel generation (f16/f32); stride-based N-D indexing; loss scaling (pragma-parsed, scale passed to backward, unscale before optimizer); FP16 master weights (master buffer allocation, init, mixed-type optimizer kernels)
5. **Compilation** — `nvcc` to shared library
6. **Execution** — PyCUDA kernel launching, memory management, loop iteration with pointer-swap rebinding, nested loop recursion, `save()` per iteration

### Key Modules

* `tensorlang/ast_builder.py` — AST construction and inlining
* `tensorlang/type_checker.py` — shape and broadcasting checks (N-D aware)
* `tensorlang/kernel_generator.py` — CUDA kernel generation
* `tensorlang/codegen.py` — code generation orchestration
* `tensorlang/compiler.py` — orchestration, execution, caching, scoped kernel names, loss scale resolution, master weight allocation
* `tensorlang/execution.py` — runtime execution, loops, rebinding
* `tensorlang/interop.py` — NumPy zero-copy interop (`InteropMixin`)
* `tensorlang/autograd.py` — NumPy reference autograd implementation (supports `scale` param, overflow detection)
* `tensorlang/fusion.py` — elementwise kernel fusion (broadcasting-aware, N-D aware)
* `tensorlang/generators/` — per-category CUDA kernel generators (base, comparisons, elementwise, nn, optimizers, reductions, shapes, unary) — all dtype-aware and N-D aware
* `tensorlang/test_runner.py` — test discovery and execution
* `tensorlang/app_runner.py` — application discovery and execution
* `tensorlang/gpu.py` — GPU detection and process-level pinning
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
source build.sh
source build.sh --test --filter autograd
source build.sh --test --filter for_loop
source build.sh --test --filter inline
```

### Test Annotations

Expected results:

```text
// @EXPECTED
// {"loss_final": 0.0164794921875}
```

Negative tests:

```text
// @EXPECT_FAILURE
```

`@EXPECT_FAILURE` currently passes if the compiler exits non-zero. It does not verify the specific failure reason.

### Tensor Verification

```bash
python3 tensorlang.py --test --verify-tensors
```

Tensor verification is performed by the test runner, not the compiler. It checks NaN/Inf, extreme values, expected values, and cached `.npy` files.

### Test Inventory

The suite covers:

* Basics, arithmetic, broadcasting (N-D)
* Activations (relu, sigmoid, tanh, softmax, gelu, swish)
* Reductions, slicing, reshape, transpose, concat (N-D)
* Linear layers, MLP, neural nets
* Normalisations (layer/batch/instance, N-D)
* Losses (mse, cross-entropy, N-D)
* Autograd (basic, chain, broadcast, relu, sigmoid, tanh, weight update, linear, layer_norm, cross_entropy, gelu, swish, reshape, transpose, slice, concat, max, min, argmax, greater, batched matmul)
* Training loops (`for_loop_*`, `training_pipeline`, `linear_regression`, …)
* Functions, pipelines, portfolio-style examples
* Load/save
* Mixed precision (f16 basic, matmul, autograd, training)
* N-dimensional (elementwise, reductions, matmul, linear, slice, concat, softmax, layer_norm, cross_entropy)
* Loss scaling (basic, FP16 upcast, auto-detect, dynamic, training loop)
* FP16 master weights (Adam optimizer)
* For-loop body parity (`for_loop_body_parity.tl`)

Full list is available via `ls tests/`. A plain-Python unittest suite also lives under `tests/python/`.

### Development Tools

Two GPU-free helpers are available for quick checks:

```bash
# Parse + typecheck any .tl file (no GPU needed)
python3 check.py path/to/file.tl

# NumPy-interpret a .tl AST (no GPU needed)
python3 verify.py path/to/file.tl
```

### Multi-GPU Test Execution

```bash
python3 tensorlang.py --list-gpus
python3 tensorlang.py --test
python3 tensorlang.py --test --gpu 1
python3 tensorlang.py --no-gpu-pinning
```

This is process-level pinning only. A single TensorLang program does not span multiple GPUs. Round-robin pinning is real but modest on the small/fast correctness suite; heavier concurrent GPU workloads are the better use case.

---

## Applications

Applications live under `apps/` and are discovered by `app.toml`. The shared toolkit `apps/tlkit/` (`chunked_runner.py`, `sequence_player.py`, `colormap.py`) supports visual/interactive apps.

```bash
python3 tensorlang.py --list-apps
python3 tensorlang.py --app examples/hello_mlp
python3 tensorlang.py --app examples/decision_boundary
python3 tensorlang.py --app games/2048 --app-args <16 numbers>
```

### App Layout

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

### App Requirements

`app.toml` may declare:

```toml
[requirements]
gpus = 1
packages = ["pygame"]
```

* `requirements.gpus` is enforced via least-utilized GPU selection and process pinning.
* `requirements.packages` are auto-installed if missing.
* `cuda_version` and `memory_gb` are display-only.
* `[gpu] device = N` can pin an app to a specific GPU.

### App Tests

```bash
python3 tensorlang.py --app examples/hello_mlp --test
```

App tests run sequentially.

### Notable Apps

* `examples/hello_mlp` — minimal 2-layer MLP training example.
* `examples/decision_boundary` — trains a 2-8-1 tanh/sigmoid MLP on XOR and animates the decision boundary. Verified: loss dropped from 0.243585 to 0.001573 over 1000 epochs.
* `games/tic_tac_toe` — complete.
* `games/2048` — tensor-ops game engine (`step.tl`) verified.

---

## Future Work

### v0.5.3 (in progress)

* **Control flow execution:** implement execution for the staged `if`/`elif`/`else` AST.
* **Autograd wiring for `batch_norm` and `instance_norm`.**
* **Single backward pass limitation:** support multiple `backward()` statements in one file.
* **Remaining loop-body gaps:** user-defined functions and file I/O other than `save()`.

### Short Term

* Improve shape-mismatch error messages
* Implement or remove `if`, `any()`, and `all()` (targeted post-v0.5)
* `batch_norm` gamma/beta affine parameters
* Dynamic loss scaling tuning (growth interval, overflow threshold)

### Medium Term

* GPU-side zero-copy interop (CUDA unified memory)
* Larger NumPy interop round-trip tests
* mmap-based zero-copy `load()` in `.tl`
* Hot reload in dev mode
* More practical example applications (vision, simple game agent, signal processing, finance) so the language does not feel purely abstract

### Long Term

* Multi-GPU single-program execution (Tier-B)
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

When adding tests, copy expected values from a real run. Do not manually round or approximate them.

---

## About

Original author: davro (David). Related projects of interest: `workspace` (PyQt6 IDE with Ollama), `we-not-me` (TradingView indicators). Primary motivation for recent work: move from pure compiler infrastructure toward practical, visible applications.

---

## License

Licensed under the GNU Lesser General Public License v3 (LGPL-3.0). See `LICENSE` for details.

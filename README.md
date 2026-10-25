Here is a concrete, audit-accurate `README.md` you can replace the current one with. It keeps the project’s strengths but stops claiming features that are parsed-only, type-checked-only, or not wired into codegen/autograd.


# TensorLang: A Native AI/ML Language

TensorLang is an open-source, native machine-learning language designed to compile tensor operations into GPU-accelerated CUDA kernels.

This project addresses pain points in Python-based ML frameworks by offering a streamlined, hardware-aware alternative for high-performance tensor computations. The compiler is implemented in Python, but the hot path — tensor operations — runs as compiled CUDA kernels.

> **Status:** v0.3.0. Core suite: 109/109 tests passing.  
> **Hardware validated on:** 2× NVIDIA GTX 1080 Ti (Pascal, sm_61).  
> **Current focus:** consistency between grammar, type checker, kernel generator, compiler, autograd, and documentation.
---

## Table of Contents

* [Overview](#overview)
* [Current Status and Limitations](#current-status-and-limitations)
* [Setup](#setup)
* [Quick Start](#quick-start)
* [Language Overview](#language-overview)
* [Operations](#operations)
* [Automatic Differentiation](#automatic-differentiation)
* [Training Loops](#training-loops)
* [File I/O](#file-io)
* [Cache](#cache)
* [Architecture](#architecture)
* [Performance](#performance)
* [Testing](#testing)
* [Applications](#applications)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

TensorLang provides a unified stack for parsing, type checking, CUDA kernel generation, compilation, and execution.

**Key Benefits:**

* **Native GPU acceleration** — tensor operations compile to CUDA kernels
* **Type safety** — compile-time tensor shape and broadcasting checks
* **ML-first design** — built specifically for neural network operations
* **Automatic differentiation** — reverse-mode autograd for a defined subset of operations
* **Training loops** — `for` loops with `backward()` and pointer-swap weight rebinding
* **Inline expressions** — nested binary/unary calls are flattened into synthetic bindings
* **Extensible architecture** — clean separation of parsing, type checking, and code generation

**Important framing:** Python is the host/compiler language. Tensor operations run as native CUDA kernels, but the compiler, runtime orchestration, and test runner are Python-based.

---

## Current Status and Limitations

TensorLang is usable for small-to-medium static-shape ML programs, but several features are intentionally limited or not yet implemented.

### Implemented and working

* Tensor declarations with explicit shapes: `Tensor[f32, (batch, features)]`
* Element-wise arithmetic with broadcasting
* Matrix multiplication (`matmul`)
* Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `swish`
* Reductions: `sum`, `mean`, `min`, `max`, `argmin`, `argmax`
* Comparisons: `greater`, `less`, `equal`
* Slicing, reshape, transpose, concat
* Normalisations: `layer_norm`, `batch_norm`, `instance_norm`
* Losses: `mse_loss`, `cross_entropy`
* Linear layers: `linear`
* Reverse-mode autograd for a defined subset of operations
* Training loops with weight rebinding
* User-defined functions
* File I/O: `load()`, `save()`
* Comments: `//` and `/* */`

### Known limitations

* **Control flow:** `if` / `elif` / `else` is present in the grammar but is **not currently built into the AST or executed**.
* **`any()` / `all()` reductions:** present in the grammar but **not implemented**.
* **`f64`:** parsed by the grammar, but type checking and kernels effectively force `f32`.
* **Autograd coverage is limited** (see [Automatic Differentiation](#automatic-differentiation)).
* **For-loop bodies support only a subset of operations** (see [Training Loops](#training-loops)).
* **Most kernels are 1D/2D only.** Higher-rank tensors are not generally supported.
* **Axis support is limited** for reductions and normalisations (commonly full reduction or axis `0`/`1` on 2D tensors).
* **`transpose(..., axes=...)`** is currently ignored; only default 2D transpose is generated.
* **`argmin` without an axis** is not supported.
* **Scalar `(1,1)` broadcasting** type-checks but currently fails during CUDA code generation.
* **Broadcast `equal`** has a code-generation dispatch bug.
* **`cross_entropy`** expects probability tensors, not logits, despite the name.
* **`batch_norm`** uses running mean/variance; it does **not** apply `gamma`/`beta` affine parameters.
* **Dev mode hot reload** is not implemented.
* **`--transpile` (WGSL)** is exposed but not currently functional.

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

# Install dependencies
bash build.sh --install

# Activate the venv
source python-env/bin/activate

# Run the full test suite
source build.sh
```

---

## Quick Start

```bash
# Compile and run a single file
python3 tensorlang.py tests/activation_pipeline.tl

# Debug mode
python3 tensorlang.py tests/inline_expr_basic.tl --debug

# Run tests
python3 tensorlang.py --test
python3 tensorlang.py --test --filter autograd
```

---

## Language Overview

### Tensor Declarations

```text
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad
```

Shapes are part of the type system. The compiler checks shape compatibility and broadcasting rules at compile time.

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
| Slicing | Supported | 1D/2D, NumPy-style ranges and indices |
| `reshape(tensor, shape)` | Supported | Element count must match |
| `transpose(tensor)` | Partial | 2D only; `axes=` is currently ignored |
| `concat(a, b, axis=...)` | Partial | 2 tensors, 2D, axis `0` or `1` |

### Linear Algebra

| Operation | Status | Notes |
|---|---|---|
| `matmul(A, B)` | Supported | 2D×2D or 2D×1D |
| `add`, `minus`, `mult`, `div` | Supported | Broadcasting supported |
| `linear(input, weight, bias)` | Supported | 1D or 2D input |

### Activations

| Operation | Status | Gradient |
|---|---|---|
| `relu` | Supported | Yes |
| `sigmoid` | Supported | Yes |
| `tanh` | Supported | Yes |
| `softmax` | Supported | Yes (1D or 2D axis `1`) |
| `gelu` | Supported | No |
| `swish` | Supported | No |

### Reductions

| Operation | Status | Notes |
|---|---|---|
| `sum` | Supported | Full or 2D axis `0`/`1` |
| `mean` | Supported | Full or 2D axis `0`/`1` |
| `min`, `max` | Supported | Full or 2D axis `0`/`1` |
| `argmin` | Partial | 2D axis `1` only |
| `argmax` | Partial | Full reduction is unreliable; 2D axis `1` supported |

### Normalisations

| Operation | Status | Notes |
|---|---|---|
| `layer_norm` | Partial | 1D or 2D axis `0`/`1`; `axis=0` has a known bug |
| `batch_norm` | Partial | 2D; uses running mean/variance; no gamma/beta |
| `instance_norm` | Partial | 2D only |

### Losses

| Operation | Status | Notes |
|---|---|---|
| `mse_loss` | Supported | Predictions and targets must have the same shape |
| `cross_entropy` | Partial | Expects probability tensors and class indices; not wired into autograd |

### Comparisons

| Operation | Status | Notes |
|---|---|---|
| `greater`, `less`, `equal` | Partial | Equal shapes or limited 2D/1D broadcast; broadcast `equal` has a dispatch bug |

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

Gradients are also saved as `*.grad.npy` when `--cache-layers` is used.

### Supported Gradient Operations

| Operation | Gradient |
|---|---|
| `matmul(A, B)` | 2D only |
| `add(A, B)` | Broadcast-aware reduction |
| `minus(A, B)` | Broadcast-aware reduction |
| `mult(A, B)` | Broadcast-aware reduction |
| `div(A, B)` | Broadcast-aware reduction |
| `relu` | Yes |
| `sigmoid` | Yes |
| `tanh` | Yes |
| `softmax` | Yes |
| `sum` | Yes |
| `mean` | Yes |
| `mse_loss` | Yes |

### Not Currently Wired into Autograd

The following operations may be present in the language but are **not currently recorded for gradient computation**:

* `linear`
* `layer_norm`
* `cross_entropy`
* `gelu`
* `swish`
* `batch_norm`
* `instance_norm`
* `concat`
* `reshape`
* `transpose`
* `slice`
* `max`, `min`
* `argmax`, `argmin`
* `greater`, `less`, `equal`

### Autograd Notes

* `matmul` gradients assume 2D operands.
* Broadcast-aware gradient reduction is implemented via `_unbroadcast()`.
* `cross_entropy` autograd is implemented in the NumPy reference but not wired into the compiler path.

---

## Training Loops

TensorLang supports multi-step gradient descent via `for` loops.

```text
for epoch in range(5) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
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

### Training Loop Constraints

* Loop counts must be integer literals: `range(5)`, not `range(n)`.
* Loop bodies support only a subset of operations.
* `save()` inside a loop is not currently executed.
* Alias nodes such as `let y = x` inside a loop are not currently resolved.
* Nested loops are not supported.
* Because all kernels are emitted into one `.cu` file, tensor names used inside a loop body cannot be reused at top level.

Example:

```text
for epoch in range(5) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    ...
}

let y_pred_final = matmul(x, w)
let loss_final = mse_loss(y_pred_final, y_true)
```

### Rebinding

```text
w = w_updated
```

This swaps GPU buffer pointers rather than copying memory.

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

If no type annotation is provided, shape is inferred from the file.

### Save

```text
save(tensor, "output.npy")
```

`save()` is currently supported at top level. It is not executed inside `for` loops.

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

The full test suite clears `cache/` before running. For isolated runs after modifying a test, clear its cache entry manually:

```bash
rm -rf cache/tests/my_test.tl/
python3 tensorlang.py tests/my_test.tl --cache-layers
```

---

## Architecture

### Compilation Pipeline

1. **Lexing & Parsing** — Lark grammar (`tensorlang.lark`)
2. **AST Construction** — flatten inline expressions; inline user-defined functions
3. **Type Checking** — validate tensor shapes, broadcasting rules, and gradient tensor pre-registration
4. **CUDA Generation** — emit forward kernels; autograd is handled by the NumPy reference graph plus compiler-recorded operations
5. **Compilation** — `nvcc` to shared library
6. **Execution** — PyCUDA kernel launching, memory management, and loop iteration with pointer-swap rebinding

### Key Modules

* `tensorlang/ast_builder.py` — AST construction and inlining
* `tensorlang/type_checker.py` — shape and broadcasting checks
* `tensorlang/kernel_generator.py` — CUDA kernel generation
* `tensorlang/compiler.py` — orchestration, execution, caching
* `tensorlang/autograd.py` — NumPy reference autograd implementation
* `tensorlang/test_runner.py` — test discovery and execution
* `tensorlang/app_runner.py` — application discovery and execution
* `tensorlang/gpu.py` — GPU detection and process-level pinning

---

## Performance

Current performance characteristics:

* Matrix multiplication uses a straightforward CUDA kernel, not cuBLAS or a tiled/shared-memory implementation.
* Training-loop weight updates use pointer swaps (no GPU memory copy between iterations).
* Generated CUDA source is cached and reused when unchanged.
* 4096×4096 matrix multiplication runs, but has not been benchmarked against cuBLAS or PyTorch.

Performance claims should be treated as unverified until baselined on the same hardware.

---

## Testing

### Single File

```bash
python3 tensorlang.py tests/for_loop_basic.tl
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

### Multi-GPU Test Execution

```bash
python3 tensorlang.py --list-gpus
python3 tensorlang.py --test
python3 tensorlang.py --test --gpu 1
python3 tensorlang.py --test --no-gpu-pinning
```

This is process-level pinning only. A single TensorLang program does not span multiple GPUs.

---

## Applications

Applications live under `apps/` and are discovered by `app.toml`.

```bash
python3 tensorlang.py --list-apps
python3 tensorlang.py --app examples/hello_mlp
python3 tensorlang.py --app examples/decision_boundary
python3 tensorlang.py --app games/2048 --app-args <16 numbers>
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

---

## Future Work

### Short Term

* Scope kernel names to remove post-loop naming constraints
* Improve shape-mismatch error messages
* Fix scalar `(1,1)` broadcast code generation
* Fix broadcast `equal` dispatch
* Fix `transpose` axes support
* Fix `layer_norm` axis `0`
* Expand autograd coverage
* Expand for-loop body operation coverage
* Fix rebind GPU memory leak

### Medium Term

* Built-in optimisers (SGD, Adam)
* Kernel fusion
* Buffer pooling for long training runs
* Nested loops
* Hot reload in dev mode
* Zero-copy Python/PyTorch interop

### Long Term

* Multi-GPU single-program execution (Tier-B)
* Mixed precision (FP16/FP32), validated on appropriate hardware
* MLIR integration
* Additional hardware backends (ROCm, Metal)

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

## License

Licensed under the GNU Lesser General Public License v3 (LGPL-3.0). See `LICENSE` for details.
---

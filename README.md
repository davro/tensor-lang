# TensorLang: A Native AI/ML Language

TensorLang is an open-source, native machine learning (ML) language designed to compile tensor operations directly into GPU-accelerated kernels. This project addresses pain points in Python-based ML frameworks by offering a streamlined, hardware-aware alternative for high-performance tensor computations.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Operations](#operations)
* [Automatic Differentiation](#automatic-differentiation)
* [Setup](#setup)
  + [System Requirements](#system-requirements)
  + [Clone](#clone)
  + [Build](#build)
* [Tests](#tests)
* [Cache](#cache)
* [Code Examples](#code-examples)
* [Architecture](#architecture)
* [Performance](#performance)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

TensorLang eliminates Python bottlenecks by providing a unified stack for parsing, type checking, and GPU code generation. The language compiles directly to optimized CUDA kernels, enabling native GPU acceleration for machine learning workloads — including forward passes, loss computation, and gradient-based weight updates.

**Key Benefits:**

* **Native GPU acceleration** - Direct CUDA kernel generation
* **Type safety** - Compile-time tensor shape and type checking
* **Zero Python overhead** - Native tensor operations without interpreter bottlenecks
* **ML-first design** - Built specifically for neural network operations
* **Automatic differentiation** - Reverse-mode autograd with broadcast-aware gradient reduction
* **Extensible architecture** - Clean separation of parsing, type checking, and code generation

## Features

* **Tensor Declarations**: Explicit tensor types with shape inference (`Tensor[f32, (batch, features)]`)
* **Comprehensive Operations**: 40+ operations covering linear algebra, activations, reductions, and comparisons
* **GPU Acceleration**: Compiles to optimized CUDA kernels with shared memory and broadcasting
* **Automatic Differentiation**: Reverse-mode autograd with `with grad` tagging and `backward()` — gradients flow through `matmul`, `add`, `mse_loss`, and more
* **Neural Network Support**: Build complete MLPs, classification networks, and feature extractors
* **Batch Processing**: Efficient handling of mini-batch training and inference
* **Memory Management**: Automatic GPU memory allocation and cleanup
* **Pipeline Operations**: Chain complex operations (slice → activate → reduce)
* **Control Flow**: `if` / `elif` / `else` conditional statements
* **User-Defined Functions**: `fn` definitions with typed parameters and return values
* **File I/O**: `load()` and `save()` for tensor persistence
* **Comment Support**: Single-line (`//`) and multi-line (`/* */`) comments
* **Comprehensive Testing**: 51+ test cases with expected results validation

## Operations

### **Tensor Creation & Manipulation**

* **Literals**: `[[1.0, 2.0], [3.0, 4.0]]` - Multi-dimensional tensor literals
* **Fill**: `fill(0.0, (3, 3))` - Create tensors with constant values
* **Slicing**: `tensor[1:3, :]` - Extract subtensors with NumPy-style syntax
* **Reshape**: `reshape(tensor, (4, 1))` - Change tensor dimensions
* **Transpose**: `transpose(tensor)` - Axis permutation with optional `axes=` argument
* **Concat**: `concat(a, b, axis=0)` - Concatenate along a dimension

### **Linear Algebra**

* **Matrix Multiplication**: `matmul(A, B)` - Optimised GPU matrix multiplication
* **Element-wise Operations**: `add`, `minus`, `mult`, `div` with broadcasting support
* **Linear Layers**: `linear(input, weight, bias)` - Complete neural network layers

### **Activation Functions**

* **ReLU**: `relu(x)` - Rectified Linear Unit activation
* **GELU**: `gelu(x)` - Gaussian Error Linear Unit
* **Swish**: `swish(x)` - Self-gated activation
* **Sigmoid**: `sigmoid(x)` - Logistic activation function
* **Tanh**: `tanh(x)` - Hyperbolic tangent activation
* **Softmax**: `softmax(x, axis=1)` - Normalised exponential with numerical stability

### **Reductions**

* **Statistical**: `sum`, `mean` - Along specified axes or full tensor
* **Min/Max**: `min`, `max` - Find minimum/maximum values
* **Argmin/Argmax**: `argmin`, `argmax` - Locate indices of extreme values

### **Normalisations**

* **Layer Norm**: `layer_norm(x, axis=1, eps=1e-5)`
* **Batch Norm**: `batch_norm(x, gamma, beta, eps=1e-5)`
* **Instance Norm**: `instance_norm(x, eps=1e-5)`

### **Loss Functions**

* **Cross Entropy**: `cross_entropy(logits, targets)`
* **MSE Loss**: `mse_loss(predictions, targets)`

### **Comparisons**

* **Element-wise**: `greater`, `less`, `equal` - Boolean operations returning 0.0/1.0
* **Broadcasting**: Support different tensor shapes following NumPy rules
* **Masking**: Create conditional masks for data filtering

---

## Automatic Differentiation

TensorLang implements reverse-mode automatic differentiation. Tensors marked `with grad` accumulate gradients during `backward()`, which traverses the computation graph in reverse and computes gradients via the chain rule.

### Syntax

```
// Mark a tensor as requiring gradient tracking
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

// Run the forward pass
let y    = matmul(x, w)
let loss = sum(y)

// Compute gradients — populates w.grad
backward(loss)
```

### Accessing Gradients

After `backward()`, gradients are available in two equivalent ways:

```
// Dot notation (attribute style)
// w.grad is saved as w.grad.npy in the cache

// Synthesised variable (usable in subsequent expressions)
let update = mult(learning_rate, w_grad)
let w_new  = minus(w, update)
```

### Supported Gradient Operations

| Operation | Gradient |
|-----------|----------|
| `matmul(A, B)` | `dL/dA = dL/dY @ B.T`, `dL/dB = A.T @ dL/dY` |
| `add(A, B)` | `dL/dA`, `dL/dB` with broadcast reduction |
| `sum(x)` | Ones tensor broadcast to input shape |
| `mse_loss(pred, target)` | `2/N * (pred - target)` |
| `linear(x, w, b)` | Gradients for `w` and `b` |

Broadcast-aware gradient reduction is implemented via `_unbroadcast()`, so bias gradients sum correctly over the batch dimension.

### Autograd Examples

**Basic gradient — matmul + sum:**
```
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

let y    = matmul(x, w)
let loss = sum(y)
backward(loss)
// w.grad = [[4.0, 4.0], [6.0, 6.0]]
```

**Chain rule — gradient through two matmuls:**
```
let x:  Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w1: Tensor[f32, (2, 2)] = [[1.0, 0.0], [0.0, 1.0]] with grad
let w2: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

let h    = matmul(x, w1)
let y    = matmul(h, w2)
let loss = sum(y)
backward(loss)
// w1.grad and w2.grad both computed correctly via chain rule
```

**Broadcast gradients — bias across a batch:**
```
let a: Tensor[f32, (3, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
let b: Tensor[f32, (2,)]   = [0.5, 0.5] with grad

let y    = add(a, b)
let loss = sum(y)
backward(loss)
// b.grad = [3.0, 3.0]  (summed over the 3-row broadcast dimension)
```

**Single SGD weight update step:**
```
let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f32, (1, 1)] = [[0.5]] with grad

let y_pred    = matmul(x, w)
let loss      = mse_loss(y_pred, y_true)
backward(loss)

let lr        = [[0.1]]
let w_updated = minus(w, mult(lr, w_grad))

let y_pred_new = matmul(x, w_updated)
let loss_after = mse_loss(y_pred_new, y_true)
// loss_after < loss  (16.875 → 4.21875)
```

---

## Setup

Clone the repository and prepare the environment to build and run TensorLang:

### System Requirements

```
sudo apt install nvidia-cuda-toolkit python3-dev
sudo apt install python3.12-venv
pip install pycuda lark numpy
```

### Clone

```
git clone https://github.com/davro/tensor-lang.git
cd tensor-lang
```

### Build

```
# Ensure the build script is executable
chmod +x build.sh

# Activate the environment
source build.sh
```

## Tests

### Single Test Execution

```
# Run individual TensorLang programs
python3 tensorlang.py tests/mlp_network.tl
python3 tensorlang.py tests/linear_classification.tl
```

### Full Test Suite

```
# Run all tests with validation
python3 tensorlang.py --test

# Run tests in parallel (default: 4 jobs)
python3 tensorlang.py --test --jobs 8

# Filter to a specific group
python3 tensorlang.py --test --filter autograd
```

**Test Coverage:**

* **Basic Operations**: Element-wise arithmetic, matrix multiplication
* **Activation Functions**: ReLU, GELU, Swish, sigmoid, tanh, softmax
* **Reductions**: Sum, mean, min, max, argmin, argmax
* **Comparisons**: Greater, less, equal with broadcasting
* **Slicing**: 1D/2D tensor slicing with range specifications
* **Linear Layers**: 1D and batch processing with bias
* **Normalisations**: Layer norm, batch norm, instance norm
* **Loss Functions**: MSE loss, cross entropy
* **Autograd**: Gradient tracking, chain rule, broadcast reduction, weight update
* **Neural Networks**: Complete MLP construction and classification
* **Pipeline Operations**: Multi-stage tensor transformations
* **Performance**: Large tensor operations (4096×4096 matrices)

### Autograd Tests

```
python3 tensorlang.py --test --filter autograd
```

```
================================================================================
State  TestCase                    Time
--------------------------------------------------------------------------------
PASS   autograd_broadcast.tl       [0.0s → 3.8s]
PASS   autograd_basic.tl           [0.0s → 3.8s]
PASS   autograd_chain.tl           [0.0s → 3.9s]
PASS   autograd_weight_update.tl   [0.0s → 3.9s]
================================================================================
Summary: 4/4 tests passed in 3.88s
```

## Cache

TensorLang stores all compilation artifacts in a cache directory organised by input file:

### Cache Structure (`cache/tests/program.tl/`)

* **`kernel.cu`**: Generated CUDA source code with optimised kernels
* **`kernel.so`**: Compiled shared library for GPU execution
* **`tensor_name.npy`**: NumPy arrays containing computed tensor results
* **`tensor_name.grad.npy`**: Gradient arrays for tensors declared `with grad`
* **Logs**: Detailed compilation and execution information

### Cache Benefits

* **Reusability**: Compiled kernels can be loaded directly in Python
* **Debugging**: Inspect generated CUDA code, intermediate results, and gradients
* **Integration**: Use TensorLang computations in existing Python workflows
* **Performance**: Skip recompilation for repeated executions

## Architecture

### Compilation Pipeline

1. **Lexing & Parsing**: Lark-based grammar with comprehensive syntax support
2. **AST Construction**: Build abstract syntax tree with operation dependencies
3. **Type Checking**: Validate tensor shapes, broadcasting rules, and operation compatibility
4. **CUDA Generation**: Emit optimised forward and backward GPU kernels
5. **Compilation**: `nvcc` compilation to shared libraries
6. **Execution**: PyCUDA-based kernel launching with memory management and gradient accumulation

### Design Principles

* **Type Safety**: Compile-time shape checking prevents runtime errors
* **GPU-First**: All operations generate native CUDA kernels — including gradient kernels
* **Composability**: Operations chain naturally with automatic memory management
* **Performance**: Optimised kernels with broadcasting, shared memory, and numerical stability
* **Extensibility**: Clean separation enables easy addition of new operations and their gradients

## Performance

**Benchmark Results** (NVIDIA GPU):

* **Matrix Multiplication**: 4096×4096 matrices execute efficiently
* **Batch Processing**: Linear layers handle large batch sizes
* **Memory Efficiency**: Automatic GPU memory management with cleanup
* **Test Suite**: 51+ tests complete in ~2 minutes including compilation

**Optimisation Features:**

* **Shared Memory**: Reduction operations use efficient parallel patterns
* **Broadcasting**: Hardware-accelerated element-wise operations with gradient reduction
* **Kernel Fusion**: Potential for combining operations (future work)
* **Numerical Stability**: Softmax uses max subtraction, comparisons use tolerance

## Code Examples

### Single SGD Training Step

```
// Linear regression: y = 2x, learning from data
let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]

let w: Tensor[f32, (1, 1)] = [[0.5]] with grad

let y_pred = matmul(x, w)
let loss   = mse_loss(y_pred, y_true)
backward(loss)

let lr        = [[0.1]]
let w_updated = minus(w, mult(lr, w_grad))
```

### Neural Network Inference

```
// 2-layer MLP for binary classification
let input: Tensor[f32, (batch, 784)] = load("input.npy")

// Hidden layer: 784 -> 256 features
let w1: Tensor[f32, (784, 256)] = load("w1.npy")
let b1: Tensor[f32, (256,)]     = load("b1.npy")
let hidden = relu(linear(input, w1, b1))

// Output layer: 256 -> 2 classes
let w2: Tensor[f32, (256, 2)] = load("w2.npy")
let b2: Tensor[f32, (2,)]     = load("b2.npy")
let logits      = linear(hidden, w2, b2)
let probs       = softmax(logits, axis=1)
let predictions = argmax(probs, axis=1)
```

### Data Processing Pipeline

```
// Feature extraction and normalisation
let raw_data: Tensor[f32, (1000, 50)] = load("features.npy")

// Remove negatives using masking
let zeros    = fill(0.0, (1000, 50))
let mask     = greater(raw_data, zeros)
let filtered = mult(raw_data, mask)

// Normalise by column means
let col_means  = mean(filtered, axis=0)
let centered   = minus(filtered, col_means)
```

## Future Work

### Short Term

* **Training Loops**: `for` loop construct to drive multi-step gradient descent natively in the language
* **Autograd Coverage**: Gradient kernels for activation functions (`relu`, `sigmoid`, `softmax`, `cross_entropy`)
* **Inline Expressions**: Allow nested calls like `relu(linear(x, w, b))` without intermediate `let` bindings
* **Error Messages**: Descriptive shape mismatch errors from the type checker

### Medium Term

* **Control Flow**: Loop constructs for training iteration
* **Custom Functions**: User-defined reusable layers
* **Optimisers**: Built-in SGD, Adam update rules
* **Memory Optimisation**: Memory pooling and buffer reuse strategies

### Long Term

* **Multi-GPU**: Distributed tensor operations
* **Mixed Precision**: FP16/FP32 automatic casting
* **MLIR Integration**: Leverage compiler infrastructure for additional backends
* **Hardware Backends**: Support for other accelerators (ROCm, Metal, etc.)

## Contributing

Contributions welcome! TensorLang is designed for extensibility:

* **New Operations**: Add grammar rules, type checking, forward kernels, and backward gradient kernels
* **Optimisations**: Improve existing kernel implementations
* **Testing**: Add test cases with `@EXPECTED` annotations for new operations
* **Documentation**: Improve examples and architectural documentation

See `CONTRIBUTING.md` for detailed guidelines.

## License

Licensed under GNU Lesser General Public License v3 (LGPL-3.0) - see `LICENSE` file for details.

---

**TensorLang** - Native ML language with GPU acceleration and automatic differentiation. Built for performance, designed for productivity.
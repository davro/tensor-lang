# TensorLang: A Native AI/ML Language

TensorLang is an open-source, native machine learning (ML) language designed to compile tensor operations directly into GPU-accelerated kernels. This project addresses pain points in Python-based ML frameworks by offering a streamlined, hardware-aware alternative for high-performance tensor computations.

---
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Operations](#operations)
- [Setup](#setup)
  - [System Requirements](#system-requirements)
  - [Clone](#clone)
  - [Build](#build)
- [Tests](#tests)
- [Cache](#cache)
- [Code Examples](#code-examples)
- [Architecture](#architecture)
- [Performance](#performance)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
TensorLang eliminates Python bottlenecks by providing a unified stack for parsing, type checking, and GPU code generation. The language compiles directly to optimized CUDA kernels, enabling native GPU acceleration for machine learning workloads.

**Key Benefits:**
- **Native GPU acceleration** - Direct CUDA kernel generation
- **Type safety** - Compile-time tensor shape and type checking
- **Zero Python overhead** - Native tensor operations without interpreter bottlenecks
- **ML-first design** - Built specifically for neural network operations
- **Extensible architecture** - Clean separation of parsing, type checking, and code generation

## Features
- **Tensor Declarations**: Explicit tensor types with shape inference (`Tensor[f32, (batch, features)]`)
- **Comprehensive Operations**: 40+ operations covering linear algebra, activations, reductions, and comparisons
- **GPU Acceleration**: Compiles to optimized CUDA kernels with shared memory and broadcasting
- **Neural Network Support**: Build complete MLPs, classification networks, and feature extractors
- **Batch Processing**: Efficient handling of mini-batch training and inference
- **Memory Management**: Automatic GPU memory allocation and cleanup
- **Pipeline Operations**: Chain complex operations (slice → activate → reduce)
- **Comment Support**: Single-line (`//`) and multi-line (`/* */`) comments
- **Comprehensive Testing**: 47+ test cases with expected results validation

## Operations

### **Tensor Creation & Manipulation**
- **Literals**: `[[1.0, 2.0], [3.0, 4.0]]` - Multi-dimensional tensor literals
- **Fill**: `fill(0.0, (3, 3))` - Create tensors with constant values
- **Slicing**: `tensor[1:3, :]` - Extract subtensors with NumPy-style syntax

### **Linear Algebra**
- **Matrix Multiplication**: `matmul(A, B)` - Optimized GPU matrix multiplication
- **Element-wise Operations**: `add`, `minus`, `mult`, `div` with broadcasting support
- **Linear Layers**: `linear(input, weight, bias)` - Complete neural network layers

### **Activation Functions**
- **ReLU**: `relu(x)` - Rectified Linear Unit activation
- **Sigmoid**: `sigmoid(x)` - Logistic activation function  
- **Tanh**: `tanh(x)` - Hyperbolic tangent activation
- **Softmax**: `softmax(x, axis=1)` - Normalized exponential with numerical stability

### **Reductions**
- **Statistical**: `sum`, `mean` - Along specified axes or full tensor
- **Min/Max**: `min`, `max` - Find minimum/maximum values
- **Argmin/Argmax**: `argmin`, `argmax` - Locate indices of extreme values

### **Comparisons**
- **Element-wise**: `greater`, `less`, `equal` - Boolean operations returning 0.0/1.0
- **Broadcasting**: Support different tensor shapes following NumPy rules
- **Masking**: Create conditional masks for data filtering

### **Pipeline Examples**
```tensorlang
// Neural network layer
let hidden = relu(linear(input, w1, b1))
let output = softmax(linear(hidden, w2, b2))

// Data preprocessing pipeline  
let filtered = mult(data, greater(data, zeros))  // Remove negatives
let normalized = div(filtered, sum(filtered))    // Normalize

// Batch processing
let batch = data[0:32, :]                       // Select batch
let features = batch[:, 1:10]                   // Extract features
let predictions = argmax(softmax(linear(features, weights, bias)), axis=1)
```

## Setup
Clone the repository and prepare the environment to build and run TensorLang:

### System Requirements
```bash
sudo apt install nvidia-cuda-toolkit python3-dev
sudo apt install python3.12-venv
pip install pycuda lark numpy
```

### Clone
```bash
git clone https://github.com/davro/tensor-lang.git
cd tensor-lang
```

### Build
```bash
# Ensure the build script is executable
chmod +x build.sh

# Activate the environment
source build.sh
```

## Tests

### Single Test Execution
```bash
# Run individual TensorLang programs
python3 tensorlang.py tests/mlp_network.tl
python3 tensorlang.py tests/linear_classification.tl
```

### Full Test Suite
```bash
# Run all 47 test cases with validation
python3 tests/runner.py
```

**Test Coverage:**
- **Basic Operations**: Element-wise arithmetic, matrix multiplication
- **Activation Functions**: ReLU, sigmoid, tanh, softmax
- **Reductions**: Sum, mean, min, max, argmin, argmax  
- **Comparisons**: Greater, less, equal with broadcasting
- **Slicing**: 1D/2D tensor slicing with range specifications
- **Linear Layers**: 1D and batch processing with bias
- **Neural Networks**: Complete MLP construction and classification
- **Pipeline Operations**: Multi-stage tensor transformations
- **Performance**: Large tensor operations (4096×4096 matrices)

## Cache 
TensorLang stores all compilation artifacts in a cache directory organized by input file:

### Cache Structure (`cache/tests/program.tl/`)
- **`kernel.cu`**: Generated CUDA source code with optimized kernels
- **`kernel.so`**: Compiled shared library for GPU execution
- **`tensor_name.npy`**: NumPy arrays containing computed tensor results
- **Logs**: Detailed compilation and execution information

### Cache Benefits
- **Reusability**: Compiled kernels can be loaded directly in Python
- **Debugging**: Inspect generated CUDA code and intermediate results
- **Integration**: Use TensorLang computations in existing Python workflows
- **Performance**: Skip recompilation for repeated executions

## Architecture

### Compilation Pipeline
1. **Lexing & Parsing**: Lark-based grammar with comprehensive syntax support
2. **AST Construction**: Build abstract syntax tree with operation dependencies  
3. **Type Checking**: Validate tensor shapes, broadcasting rules, and operation compatibility
4. **CUDA Generation**: Emit optimized GPU kernels with shared memory and atomic operations
5. **Compilation**: nvcc compilation to shared libraries
6. **Execution**: PyCUDA-based kernel launching with memory management

### Design Principles
- **Type Safety**: Compile-time shape checking prevents runtime errors
- **GPU-First**: All operations generate native CUDA kernels
- **Composability**: Operations chain naturally with automatic memory management
- **Performance**: Optimized kernels with broadcasting, shared memory, and numerical stability
- **Extensibility**: Clean separation enables easy addition of new operations

## Performance

**Benchmark Results** (NVIDIA GPU):
- **Matrix Multiplication**: 4096×4096 matrices execute efficiently
- **Batch Processing**: Linear layers handle large batch sizes
- **Memory Efficiency**: Automatic GPU memory management with cleanup
- **Test Suite**: 47 tests complete in ~2 minutes including compilation

**Optimization Features:**
- **Shared Memory**: Reduction operations use efficient parallel patterns
- **Broadcasting**: Hardware-accelerated element-wise operations
- **Kernel Fusion**: Potential for combining operations (future work)
- **Numerical Stability**: Softmax uses max subtraction, comparisons use tolerance

## Code Examples

### Neural Network Construction
```tensorlang
// 2-layer MLP for binary classification
let input: Tensor[f32, (batch, 784)] = load_data()

// Hidden layer: 784 -> 256 features
let hidden_w: Tensor[f32, (784, 256)] = random_weights()
let hidden_b: Tensor[f32, (256,)] = zeros()
let hidden = relu(linear(input, hidden_w, hidden_b))

// Output layer: 256 -> 2 classes
let output_w: Tensor[f32, (256, 2)] = random_weights() 
let output_b: Tensor[f32, (2,)] = zeros()
let logits = linear(hidden, output_w, output_b)
let probs = softmax(logits, axis=1)

// Predictions
let predictions = argmax(probs, axis=1)
```

### Data Processing Pipeline
```tensorlang
// Feature extraction and normalization
let raw_data: Tensor[f32, (1000, 50)] = load_features()

// Remove outliers using masking
let outlier_mask = less(abs(raw_data), fill(3.0, (1000, 50)))
let clean_data = mult(raw_data, outlier_mask)

// Normalize by column statistics  
let col_means = mean(clean_data, axis=0)
let centered = minus(clean_data, col_means)
let col_stds = sqrt(mean(mult(centered, centered), axis=0))
let normalized = div(centered, col_stds)
```

## Future Work

### Short Term
- **More Activations**: GELU, Swish, LayerNorm primitives
- **Advanced Indexing**: Boolean and fancy indexing support
- **Optimizations**: Kernel fusion for operation chains
- **Error Handling**: Better error messages and recovery

### Medium Term  
- **Automatic Differentiation**: Gradient computation for training
- **Control Flow**: Conditional operations and loops
- **Custom Functions**: User-defined operations and layers
- **Memory Optimization**: Memory pooling and reuse strategies

### Long Term
- **Multi-GPU**: Distributed tensor operations
- **Mixed Precision**: FP16/FP32 automatic casting
- **MLIR Integration**: Leverage compiler infrastructure
- **Hardware Backends**: Support for other accelerators (ROCm, Metal, etc.)

## Contributing

Contributions welcome! TensorLang is designed for extensibility:

- **New Operations**: Add grammar rules, type checking, and CUDA kernels
- **Optimizations**: Improve existing kernel implementations
- **Testing**: Add test cases for edge cases and new operations  
- **Documentation**: Improve examples and architectural documentation

See `CONTRIBUTING.md` for detailed guidelines.

## License
Licensed under GNU Lesser General Public License v3 (LGPL-3.0) - see `LICENSE` file for details.

---

**TensorLang** - Native ML language with GPU acceleration. Built for performance, designed for productivity.
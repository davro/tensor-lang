// TensorLang: 2-Layer Neural Network Forward Pass
// Input batch of 4 samples with 3 features each

// Layer 1 weights (3 inputs × 5 hidden)
let W1 : Tensor[f32, (3, 5)] = [
  [0.2, -0.1, 0.4, 0.0, 0.5],
  [-0.3, 0.8, -0.2, 0.1, -0.5],
  [0.7, -0.4, 0.3, 0.6, -0.1]
]

// Layer 1 bias (5)
let b1 : Tensor[f32, (5)] = [0.1, 0.1, 0.1, 0.1, 0.1]

// Layer 2 weights (5 hidden × 2 outputs)
let W2: Tensor[f32, (5, 2)] = [
  [0.5, -0.2],
  [-0.3, 0.1],
  [0.2, 0.7],
  [0.0, 0.5],
  [0.4, -0.6]
]

// Layer 2 bias (2)
let b2 : Tensor[f32, (2)] = [0.0, 0.0]

// Input batch (4 samples × 3 features)
let X : Tensor[f32, (4, 3)] = [
  [1.0, 0.5, -1.0],
  [-0.2, 0.3, 0.8],
  [0.0, -0.5, 0.5],
  [1.2, 0.7, -0.3]
]

// Forward pass
let h1 = matmul(X, W1)
let h1_bias = add(h1, b1)     // broadcast add
let h1_act = relu(h1_bias)

let out = matmul(h1_act, W2)
let out_bias = add(out, b2)   // broadcast add

// Final output (4 samples × 2 outputs)
out_bias


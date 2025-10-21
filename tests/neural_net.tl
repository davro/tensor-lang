// ============================================================================
// TensorLang: Neural Net
// ============================================================================
// Description: Tests neural network operations
//
// Architecture Pattern:
//   'out_bias' output
//
// Expected Output Variables:
//   out_bias
//
// @EXPECTED
// {
//   "out_bias": [
//     [
//       -0.04,
//       -0.12
//     ],
//     [
//       0.293,
//       0.343
//     ],
//     [
//       0.49,
//       0.12
//     ],
//     [
//       0.024,
//       0.083
//     ]
//   ]
// }

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


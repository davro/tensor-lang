// ============================================================================
// TensorLang: Mlp
// ============================================================================
// Description: Tests multi-layer perceptron
//
// Architecture Pattern:
//   Network output
//
// Expected Output Variables:
//   output
//
// @EXPECTED
// {
//   "output": [
//     [
//       0.22000001,
//       0.29
//     ],
//     [
//       0.70000005,
//       -0.05
//     ],
//     [
//       0.26000002,
//       0.32
//     ],
//     [
//       0.4,
//       0.29999998
//     ]
//   ]
// }

let X : Tensor[f32, (4, 3)] = [
  [1.0, 2.0, 1.0],
  [2.0, 1.0, 3.0],
  [1.0, 3.0, 2.0],
  [3.0, 2.0, 1.0]
]
// Layer 1: 3 → 4
let W1 : Tensor[f32, (3, 4)] = [
  [0.1, 0.2, -0.1, 0.3],
  [-0.2, 0.1, 0.4, -0.1],
  [0.3, -0.3, 0.2, 0.2]
]
let b1 : Tensor[f32, (4)] = [0.1, 0.0, -0.1, 0.2]
let linear_output1 = matmul(X, W1)
let linear_output_add = add(linear_output1, b1)
let h1 = relu(linear_output_add)
// Output: 4 → 2
let W2 : Tensor[f32, (4, 2)] = [
  [0.2, -0.1],
  [-0.3, 0.4],
  [0.1, 0.2],
  [0.4, -0.2]
]
let b2 : Tensor[f32, (2)] = [0.0, 0.1]
let linear_output2 = matmul(h1, W2)
let output = add(linear_output2, b2)

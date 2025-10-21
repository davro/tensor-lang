// ============================================================================
// TensorLang: Linear Regression
// ============================================================================
// Description: Tests linear transformation layer
//
// Architecture Pattern:
//   Classification output
//
// Expected Output Variables:
//   predictions
//
// @EXPECTED
// {
//   "predictions": [
//     [
//       2.1
//     ],
//     [
//       2.9
//     ],
//     [
//       3.7
//     ],
//     [
//       4.5
//     ],
//     [
//       5.3
//     ]
//   ]
// }

let X : Tensor[f32, (5, 2)] = [
  [1.0, 2.0], 
  [2.0, 3.0], 
  [3.0, 4.0], 
  [4.0, 5.0], 
  [5.0, 6.0]
]
let W : Tensor[f32, (2, 1)] = [[0.5], [0.3]]
let b : Tensor[f32, (1)] = [1.0]
let linear_output = matmul(X, W)
let predictions = add(linear_output, b)

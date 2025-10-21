// ============================================================================
// TensorLang: Comparison Broadcast
// ============================================================================
// Description: Tests broadcasting behavior
//
// Architecture Pattern:
//   'result' output
//
// Expected Output Variables:
//   result
//
// @EXPECTED
// {
//   "result": [
//     [
//       0.0,
//       0.0,
//       1.0
//     ],
//     [
//       1.0,
//       1.0,
//       1.0
//     ]
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let threshold: Tensor[f32, (3,)] = [2.0, 2.0, 2.0]
let result = greater(matrix, threshold)
result

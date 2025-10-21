// ============================================================================
// TensorLang: Reshape
// ============================================================================
// Description: Tests tensor reshaping
//
// Architecture Pattern:
//   'reshaped' output
//
// Expected Output Variables:
//   reshaped
//
// @EXPECTED
// {
//   "reshaped": [
//     [
//       1.0,
//       2.0,
//       3.0,
//       4.0
//     ],
//     [
//       5.0,
//       6.0,
//       7.0,
//       8.0
//     ],
//     [
//       9.0,
//       10.0,
//       11.0,
//       12.0
//     ]
//   ]
// }

let matrix: Tensor[f32, (2, 6)] = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
let reshaped = reshape(matrix, (3, 4))
reshaped
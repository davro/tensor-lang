// ============================================================================
// TensorLang: Transpose
// ============================================================================
// Description: Tests tensor transposition
//
// Architecture Pattern:
//   'transposed' output
//
// Expected Output Variables:
//   transposed
//
// @EXPECTED
// {
//   "transposed": [
//     [
//       1.0,
//       4.0
//     ],
//     [
//       2.0,
//       5.0
//     ],
//     [
//       3.0,
//       6.0
//     ]
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let transposed = transpose(matrix)
transposed
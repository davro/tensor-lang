// ============================================================================
// TensorLang: Slice 2D Rows
// ============================================================================
// Description: Tests tensor slicing
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
//       1.0,
//       2.0,
//       3.0
//     ],
//     [
//       4.0,
//       5.0,
//       6.0
//     ]
//   ]
// }

let matrix: Tensor[f32, (3, 3)] = [[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]]
let result = matrix[0:2, :]
result

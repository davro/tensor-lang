// ============================================================================
// TensorLang: Slice Submatrix
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
//       6.0,
//       7.0
//     ],
//     [
//       10.0,
//       11.0
//     ]
//   ]
// }

let matrix: Tensor[f32, (4, 4)] = [[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0, 16.0]]
let result = matrix[1:3, 1:3]
result

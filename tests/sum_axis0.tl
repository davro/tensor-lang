// ============================================================================
// TensorLang: Sum Axis0
// ============================================================================
// Description: Tests sum reduction
//
// Architecture Pattern:
//   'col_sums' output
//
// Expected Output Variables:
//   col_sums
//
// @EXPECTED
// {
//   "col_sums": [
//     5.0,
//     7.0,
//     9.0
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let col_sums = sum(matrix, axis=0)
col_sums

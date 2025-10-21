// ============================================================================
// TensorLang: Sum Axis
// ============================================================================
// Description: Tests sum reduction
//
// Architecture Pattern:
//   'row_sums' output
//
// Expected Output Variables:
//   row_sums
//
// @EXPECTED
// {
//   "row_sums": [
//     6.0,
//     15.0
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let row_sums = sum(matrix, axis=1)
row_sums

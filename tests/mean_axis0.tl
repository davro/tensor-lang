// ============================================================================
// TensorLang: Mean Axis0
// ============================================================================
// Description: Tests mean reduction
//
// Architecture Pattern:
//   'col_means' output
//
// Expected Output Variables:
//   col_means
//
// @EXPECTED
// {
//   "col_means": [
//     2.5,
//     3.5,
//     4.5
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let col_means = mean(matrix, axis=0)
col_means

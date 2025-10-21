// ============================================================================
// TensorLang: Mean Axis
// ============================================================================
// Description: Tests mean reduction
//
// Architecture Pattern:
//   'row_means' output
//
// Expected Output Variables:
//   row_means
//
// @EXPECTED
// {
//   "row_means": [
//     2.0,
//     5.0
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let row_means = mean(matrix, axis=1)
row_means

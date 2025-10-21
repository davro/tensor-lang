// ============================================================================
// TensorLang: Max Axis
// ============================================================================
// Description: Tests maximum reduction
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
//     4.0,
//     6.0
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[1.0, 4.0, 2.0], [5.0, 3.0, 6.0]]
let result = max(matrix, axis=1)
result

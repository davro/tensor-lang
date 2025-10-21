// ============================================================================
// TensorLang: Min Axis
// ============================================================================
// Description: Tests minimum reduction
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
//     2.0,
//     1.0,
//     1.0
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]
let result = min(matrix, axis=0)
result
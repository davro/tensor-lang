// ============================================================================
// TensorLang: Argmin
// ============================================================================
// Description: Tests argmin reduction operation
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
//     1.0,
//     2.0
//   ]
// }

let matrix: Tensor[f32, (2, 3)] = [[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]
let result = argmin(matrix, axis=1)
result

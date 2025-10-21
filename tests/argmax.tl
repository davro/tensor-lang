// ============================================================================
// TensorLang: Argmax
// ============================================================================
// Description: Tests argmax reduction operation
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

let matrix: Tensor[f32, (2, 3)] = [[1.0, 4.0, 2.0], [5.0, 3.0, 6.0]]
let result = argmax(matrix, axis=1)
result

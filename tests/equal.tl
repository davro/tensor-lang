// ============================================================================
// TensorLang: Equal
// ============================================================================
// Description: Tests equality comparison
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
//     0.0,
//     1.0
//   ]
// }

let a: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let b: Tensor[f32, (3,)] = [1.0, 2.0001, 3.0]
let result = equal(a, b)
result

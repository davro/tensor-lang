// ============================================================================
// TensorLang: Greater
// ============================================================================
// Description: Tests greater than comparison
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
//     0.0,
//     1.0,
//     1.0
//   ]
// }

let a: Tensor[f32, (3,)] = [1.0, 3.0, 5.0]
let b: Tensor[f32, (3,)] = [2.0, 2.0, 4.0]
let result = greater(a, b)
result

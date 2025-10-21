// ============================================================================
// TensorLang: Max Full
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
//   "result": 5.0
// }

let data: Tensor[f32, (5,)] = [1.0, 5.0, 2.0, 4.0, 3.0]
let result = max(data)
result

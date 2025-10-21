// ============================================================================
// TensorLang: Min Full
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
//   "result": 1.0
// }

let data: Tensor[f32, (5,)] = [3.0, 1.0, 4.0, 2.0, 5.0]
let result = min(data)
result

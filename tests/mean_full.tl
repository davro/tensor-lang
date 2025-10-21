// ============================================================================
// TensorLang: Mean Full
// ============================================================================
// Description: Tests mean reduction
//
// Architecture Pattern:
//   'average' output
//
// Expected Output Variables:
//   average
//
// @EXPECTED
// {
//   "average": 3.0
// }

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let average = mean(data)
average

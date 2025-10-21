// ============================================================================
// TensorLang: Sum Full
// ============================================================================
// Description: Tests sum reduction
//
// Architecture Pattern:
//   'total' output
//
// Expected Output Variables:
//   total
//
// @EXPECTED
// {
//   "total": 15.0
// }

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let total = sum(data)
total

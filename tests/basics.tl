// ============================================================================
// TensorLang: Basics
// ============================================================================
// Description: Tests tensor operations
//
// Architecture Pattern:
//   'total' output
//
// Expected Output Variables:
//   total
//
// @EXPECTED
// {
//   "total": [
//     10.8,
//     27.0,
//     16.2,
//     32.4
//   ]
// }

let prices : Tensor[f32, (4)] = [10.0, 25.0, 15.0, 30.0]
let tax_rate : Tensor[f32, (4)] = [0.08, 0.08, 0.08, 0.08]
let taxes = mult(prices, tax_rate)
let total = add(prices, taxes)

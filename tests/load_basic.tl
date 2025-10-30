// tests/load_basic.tl
// ============================================================================
// TensorLang: Basic Load Test
// ============================================================================
// Description: Test loading .npy files
//
// Setup: python3 scripts/generate_test_data.py
//
// @EXPECTED
// {
//   "prices": [10, 25, 15, 30],
//   "tax_rate": [0.08, 0.08, 0.08, 0.08],
//   "taxes": [0.79999995, 2, 1.1999999, 2.3999999],
//   "total": [10.8, 27.0, 16.2, 32.4]
// }

let prices:   Tensor[f32, (4)] = load("cache/tests/basics.tl/prices.npy")
let tax_rate: Tensor[f32, (4)] = load("cache/tests/basics.tl/tax_rate.npy")

let taxes = mult(prices, tax_rate)
let total = add(prices, taxes)
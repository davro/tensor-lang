
// ============================================================================
// TensorLang: Dynamic Load Test
// ============================================================================
// Description: Test loading .npy files WITHOUT explicit shape annotations
//
// This demonstrates dynamic shape inference, useful for:
//   - Data that changes size over time (daily updates)
//   - Unknown data dimensions at compile time
//   - Flexible data pipelines
//
// Depends on: tests/basics.tl
//
// @EXPECTED
// {
//   "prices": [10.0, 25.0, 15.0, 30.0],
//   "tax_rate": [0.08, 0.08, 0.08, 0.08],
//   "taxes": [0.8, 2.0, 1.2, 2.4],
//   "total": [10.8, 27.0, 16.2, 32.4]
// }

// DYNAMIC LOADING - No explicit shape required!
let prices = load("cache/tests/basics.tl/prices.npy")
let tax_rate = load("cache/tests/basics.tl/tax_rate.npy")

// These operations work because shape is inferred at load time
let taxes = mult(prices, tax_rate)
let total = add(prices, taxes)

// tests/load_inferred.tl
// Inferred shape - adapts to file

// @EXPECTED
// {
//   "prices": [10, 25, 15, 30],
//   "taxes": [0.8, 2.0, 1.2, 2.4]
// }

// No type annotation - shape inferred from file
let prices = load("cache/tests/basics.tl/prices.npy")
let tax_rate = load("cache/tests/basics.tl/tax_rate.npy")

let taxes = mult(prices, tax_rate)
let total = add(prices, taxes)

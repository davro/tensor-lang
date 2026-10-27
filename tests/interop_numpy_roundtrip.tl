// ============================================================================
// TensorLang: NumPy interop — load/save round-trip
// ============================================================================

let x = load("tests/fixtures/input.npy")
let y = add(x, x)
save(y, "output.npy")

// @EXPECTED
// {
//   "y": [[2.0, 4.0], [6.0, 8.0]]
// }

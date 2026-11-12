// tests/load_f64_inferred.tl
// ============================================================================
// TensorLang: Load of an undeclared float64 .npy
// ============================================================================
// Description: load() without annotation of a float64 file is inferred as f32
//              and converted on load (NumPy writes float64 by default).
//              Fixture: tests/load_f64_data.npy, float64, [1.5, 2.5, 4.0, 0.25]
//
// @EXPECTED
// {
//   "a": [1.5, 2.5, 4.0, 0.25],
//   "b": [3.0, 5.0, 8.0, 0.5]
// }

let a = load("tests/load_f64_data.npy")
let b = add(a, a)

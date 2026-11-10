// ============================================================================
// TensorLang diagnostics: a declared f64 tensor is rejected (it used to be misread by comparisons)
// ============================================================================
// f64 has no kernel support. Comparison kernels used to read the 8-byte data as f32 and return wrong values
// without any error.
// @EXPECT_FAILURE: [E-UNSUPPORTED-DTYPE]
// @EXPECT_FAILURE: declared as f64
let x: Tensor[f64, (3,)] = [1.0, 2.0, 3.0]
let y: Tensor[f64, (3,)] = [2.0, 2.0, 2.0]
let gt = greater(x, y)

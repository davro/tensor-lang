// ============================================================================
// TensorLang diagnostics: a declared f64 tensor is rejected (it used to fail with KeyError: 'f64')
// ============================================================================
// @EXPECT_FAILURE: [E-UNSUPPORTED-DTYPE]
// @EXPECT_FAILURE: declared as f64
let x: Tensor[f64, (3,)] = fill(1.0, (3,))
let y: Tensor[f32, (3,)] = fill(2.0, (3,))
let s = add(x, y)

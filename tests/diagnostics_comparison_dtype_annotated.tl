// ============================================================================
// TensorLang diagnostics: an annotated comparison let is checked for dtype too
// ============================================================================
// @EXPECT_FAILURE: [E-COMPARISON-DTYPE]
// @EXPECT_FAILURE: comparison kernels only support f32 data
let a: Tensor[f16, (3,)] = fill(1.0, (3,))
let b: Tensor[f16, (3,)] = fill(2.0, (3,))
let m: Tensor[f32, (3,)] = equal(a, b)

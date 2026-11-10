// ============================================================================
// TensorLang diagnostics: an f16 tensor against an f32 tensor is rejected
// ============================================================================
// @EXPECT_FAILURE: [E-COMPARISON-DTYPE]
// @EXPECT_FAILURE: (f16) with
// @EXPECT_FAILURE: (f32)
let a: Tensor[f16, (3,)] = fill(1.0, (3,))
let b: Tensor[f32, (3,)] = fill(2.0, (3,))
let m = less(a, b)

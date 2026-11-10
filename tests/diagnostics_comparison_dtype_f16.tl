// ============================================================================
// TensorLang diagnostics: a comparison of f16 tensors is rejected
// ============================================================================
// The comparison kernels are generated for f32 data only, so f16 operands would be misread.
// @EXPECT_FAILURE: [E-COMPARISON-DTYPE]
// @EXPECT_FAILURE: comparison kernels only support f32 data
let a: Tensor[f16, (3,)] = fill(1.0, (3,))
let b: Tensor[f16, (3,)] = fill(2.0, (3,))
let m = greater(a, b)

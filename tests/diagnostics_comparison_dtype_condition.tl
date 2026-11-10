// ============================================================================
// TensorLang diagnostics: an f16 tensor against a literal in a condition is rejected
// ============================================================================
// A numeric literal is an f32 fill, so `a > 0.5` would mix f16 and f32 data.
// @EXPECT_FAILURE: [E-COMPARISON-DTYPE]
// @EXPECT_FAILURE: if/elif condition
let a: Tensor[f16, (3,)] = fill(1.0, (3,))
if a > 0.5 {
    let y: Tensor[f16, (3,)] = add(a, a)
}

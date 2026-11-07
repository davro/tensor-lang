// ============================================================================
// TensorLang diagnostics: undefined tensor names the expression that uses it
// ============================================================================
// @EXPECT_FAILURE: [E-UNDEFINED-TENSOR]
// @EXPECT_FAILURE: uses 'nope'
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let y = add(x, nope)

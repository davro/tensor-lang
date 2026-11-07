// ============================================================================
// TensorLang diagnostics: a near-miss name gets a "did you mean" suggestion
// ============================================================================
// @EXPECT_FAILURE: [E-UNDEFINED-TENSOR]
// @EXPECT_FAILURE: Did you mean 'weights' instead of 'wieghts'?
let weights: Tensor[f32, (2,)] = [1.0, 2.0]
let y = add(weights, wieghts)

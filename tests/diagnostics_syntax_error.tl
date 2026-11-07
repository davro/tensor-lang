// ============================================================================
// TensorLang diagnostics: ordinary syntax errors get a short coded message
// ============================================================================
// @EXPECT_FAILURE: [E-SYNTAX]
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let = 5

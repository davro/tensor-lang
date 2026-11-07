// ============================================================================
// TensorLang diagnostics: an error inside a nested call names the expression,
// not the compiler temporary (__tmp_N)
// ============================================================================
// @EXPECT_FAILURE: [E-UNDEFINED-TENSOR]
// @EXPECT_FAILURE: the nested expression `mult(x, nope)`
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let r = add(x, mult(x, nope))

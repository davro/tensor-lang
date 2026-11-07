// ============================================================================
// TensorLang diagnostics: a bare number is not a value
// ============================================================================
// Previously `let s = 3.0` was skipped silently and only failed later, as an
// undefined tensor, where `s` was used.
// @EXPECT_FAILURE: [E-SCALAR-LITERAL]
// @EXPECT_FAILURE: let s: Tensor[f32, (1,)] = [3.0]
let s = 3.0
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let y = add(x, s)

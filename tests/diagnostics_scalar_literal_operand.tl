// ============================================================================
// TensorLang diagnostics: a numeric literal used as an operand
// ============================================================================
// The grammar only allows numeric literals in comparison conditions and
// optimizer arguments, so this is reported by the parse-error path.
// @EXPECT_FAILURE: [E-SCALAR-LITERAL]
// @EXPECT_FAILURE: declare a tensor
let x: Tensor[f32, (2,)] = [1.0, 2.0]
let y = add(x, 2.0)

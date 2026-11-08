// ============================================================================
// TensorLang: chained comparisons are rejected
// ============================================================================
// `a > b > c` parses right-associatively as a > (b > c), which is never what
// was meant, so it is a syntax error.
// @EXPECT_FAILURE: [E-SYNTAX]
// @EXPECT_FAILURE: chained comparisons
let a: Tensor[f32, (1,)] = [1.0]
let m = a > a > a

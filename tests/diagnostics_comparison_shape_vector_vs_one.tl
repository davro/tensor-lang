// ============================================================================
// TensorLang diagnostics: a comparison needs equal shapes (or a matrix against a row)
// ============================================================================
// A (3,) tensor against a (1,) tensor passes NumPy-style broadcasting, but the
// comparison kernels cannot broadcast a (1,) operand, so it is rejected up front.
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: comparison kernels support equal shapes
let a: Tensor[f32, (3,)] = fill(1.0, (3,))
let b: Tensor[f32, (1,)] = fill(2.0, (1,))
let m = greater(a, b)

// ============================================================================
// TensorLang diagnostics: an annotated comparison let is checked too
// ============================================================================
// A declared type used to make the type checker trust the result without looking at the operands.
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: comparison kernels support equal shapes
let a: Tensor[f32, (3,)] = fill(1.0, (3,))
let b: Tensor[f32, (1,)] = fill(2.0, (1,))
let m: Tensor[f32, (3,)] = greater_equal(a, b)

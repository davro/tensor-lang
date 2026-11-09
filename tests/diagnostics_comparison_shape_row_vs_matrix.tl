// ============================================================================
// TensorLang diagnostics: a row must be the SECOND operand against a matrix
// ============================================================================
// The kernels support (rows, cols) against (cols,), not (cols,) against (rows, cols).
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: comparison kernels support equal shapes
let row: Tensor[f32, (3,)] = fill(1.0, (3,))
let mat: Tensor[f32, (2, 3)] = fill(2.0, (2, 3))
let m = less(row, mat)

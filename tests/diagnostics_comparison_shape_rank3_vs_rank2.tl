// ============================================================================
// TensorLang diagnostics: a rank-3 tensor cannot be compared with a rank-2 tensor
// ============================================================================
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: comparison kernels support equal shapes
let cube: Tensor[f32, (2, 2, 3)] = fill(1.0, (2, 2, 3))
let mat: Tensor[f32, (2, 3)] = fill(2.0, (2, 3))
let m = not_equal(cube, mat)

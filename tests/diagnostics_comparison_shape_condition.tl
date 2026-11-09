// ============================================================================
// TensorLang diagnostics: an if/elif condition follows the same comparison shape rule
// ============================================================================
// @EXPECT_FAILURE: [E-SHAPE-MISMATCH]
// @EXPECT_FAILURE: comparison kernels support equal shapes
let a: Tensor[f32, (3,)] = fill(1.0, (3,))
let b: Tensor[f32, (1,)] = fill(2.0, (1,))
if a > b {
    let y: Tensor[f32, (3,)] = add(a, a)
}

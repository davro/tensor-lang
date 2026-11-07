// ============================================================================
// TensorLang diagnostics: a branch mismatch names the branches and both types
// ============================================================================
// @EXPECT_FAILURE: [E-BRANCH-MISMATCH]
// @EXPECT_FAILURE: the `if` branch defines it as f32 (2, 2)
// @EXPECT_FAILURE: the `else` branch defines it as f32 (1,)
let x: Tensor[f32, (2, 2)] = fill(1.0, (2, 2))
if x > 0.0 {
    let y: Tensor[f32, (2, 2)] = add(x, x)
} elif x < 0.0 {
    let y: Tensor[f32, (2, 2)] = mult(x, x)
} else {
    let y: Tensor[f32, (1,)] = fill(1.0, (1,))
}
let result = y

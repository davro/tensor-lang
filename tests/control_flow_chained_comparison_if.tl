// ============================================================================
// TensorLang: chained comparisons are rejected in conditions too
// ============================================================================
// @EXPECT_FAILURE: [E-SYNTAX]
// @EXPECT_FAILURE: chained comparisons
let a: Tensor[f32, (1,)] = [1.0]
let z: Tensor[f32, (1,)] = [0.0]
if a > a > a {
    let z = a
}
z

// ============================================================================
// TensorLang diagnostics: undefined tensor in an if condition
// ============================================================================
// @EXPECT_FAILURE: [E-UNDEFINED-TENSOR]
// @EXPECT_FAILURE: the condition `x > missing`
let x: Tensor[f32, (1,)] = [1.0]
let z: Tensor[f32, (1,)] = [0.0]
if x > missing {
    let z = x
}
z

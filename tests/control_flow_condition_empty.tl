// ============================================================================
// TensorLang: an empty tensor in a condition is a compile error
// ============================================================================
// A zero-sized tensor has no elements to be "all nonzero", so a condition on it
// has no meaning. (Previously it silently evaluated false at run time.)
// @EXPECT_FAILURE: [E-CONDITION-EMPTY]
// @EXPECT_FAILURE: the condition `e`
let e: Tensor[f32, (0,)] = fill(1.0, (0,))
let z: Tensor[f32, (1,)] = [0.0]
if e {
    let z = z
}
z

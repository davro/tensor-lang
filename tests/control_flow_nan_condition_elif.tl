// ============================================================================
// TensorLang: NaN in an elif condition is a run-time error
// ============================================================================
// The `if` (x > 5.0) is false, so evaluation reaches the elif, whose operand is NaN.
// @EXPECT_FAILURE: [E-NAN-IN-CONDITION]
// @EXPECT_FAILURE: the condition `ratio > 0.5`
let x: Tensor[f32, (1,)] = [1.0]
let ratio: Tensor[f32, (1,)] = load("tests/condition_nan_data.npy")
let a: Tensor[f32, (1,)] = [10.0]
let z: Tensor[f32, (1,)] = [0.0]
if x > 5.0 {
    let z = a
} elif ratio > 0.5 {
    let z = a
}
z

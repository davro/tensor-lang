// ============================================================================
// TensorLang: NaN inside any() in a condition is a run-time error
// ============================================================================
// @EXPECT_FAILURE: [E-NAN-IN-CONDITION]
// @EXPECT_FAILURE: the condition `any(ratio)`
let ratio: Tensor[f32, (1,)] = load("tests/condition_nan_data.npy")
let a: Tensor[f32, (1,)] = [10.0]
let z: Tensor[f32, (1,)] = [0.0]
if any(ratio) {
    let z = a
}
z

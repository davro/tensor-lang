// ============================================================================
// TensorLang: NaN as a bare-name condition is a run-time error
// ============================================================================
// (A bare NaN would otherwise count as "nonzero" and take the true branch.)
// @EXPECT_FAILURE: [E-NAN-IN-CONDITION]
// @EXPECT_FAILURE: 'ratio' contains NaN
let ratio: Tensor[f32, (1,)] = load("tests/condition_nan_data.npy")
let a: Tensor[f32, (1,)] = [10.0]
let z: Tensor[f32, (1,)] = [0.0]
if ratio {
    let z = a
}
z

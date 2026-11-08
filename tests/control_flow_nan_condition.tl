// ============================================================================
// TensorLang: NaN in a comparison condition is a run-time error
// ============================================================================
// Every comparison with NaN is false (except !=), so without this check the
// program would silently take the else branch. The NaN is loaded from a fixture
// (tests/condition_nan_data.npy holds [nan]) so the test does not depend on how
// division by zero behaves.
// @EXPECT_FAILURE: [E-NAN-IN-CONDITION]
// @EXPECT_FAILURE: 'ratio' contains NaN
// @EXPECT_FAILURE: the condition `ratio > limit`
let ratio: Tensor[f32, (1,)] = load("tests/condition_nan_data.npy")
let limit: Tensor[f32, (1,)] = [1.0]
let z: Tensor[f32, (1,)] = [0.0]
if ratio > limit {
    let z = limit
} else {
    let z = ratio
}
z

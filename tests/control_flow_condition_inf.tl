// ============================================================================
// TensorLang: infinity is a legal condition operand (only NaN is rejected)
// ============================================================================
// inf > 1.0 is true, so z = a. Fixture tests/condition_inf_data.npy holds [inf].
// Expected value derived by hand.
//
// @EXPECTED
// {"z": [10.0]}
let big: Tensor[f32, (1,)] = load("tests/condition_inf_data.npy")
let limit: Tensor[f32, (1,)] = [1.0]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let z: Tensor[f32, (1,)] = [0.0]
if big > limit {
    let z = a
} else {
    let z = b
}
z

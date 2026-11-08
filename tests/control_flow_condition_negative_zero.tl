// ============================================================================
// TensorLang: -0.0 is falsy (it equals zero)
// ============================================================================
// A bare-name condition is true only if every element is nonzero; -0.0 == 0, so
// the else branch runs and z = b. Expected value derived by hand.
//
// @EXPECTED
// {"z": [20.0]}
let nz: Tensor[f32, (1,)] = [-0.0]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let z: Tensor[f32, (1,)] = [0.0]
if nz {
    let z = a
} else {
    let z = b
}
z

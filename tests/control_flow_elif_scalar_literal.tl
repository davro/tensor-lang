// ============================================================================
// TensorLang: scalar literals in elif conditions (elif branch taken)
// ============================================================================
// Regression: elif conditions were never flattened, so a bare literal such as
// `x > 0.5` produced an unregistered synthetic binding. x=1.0, so the `if`
// (x > 5.0) is false and the `elif` (x > 0.5) is true -> z = b.
// Expected value derived by hand.
//
// @EXPECTED
// {"z": [20.0]}
let x: Tensor[f32, (1,)] = [1.0]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let c: Tensor[f32, (1,)] = [30.0]
let z: Tensor[f32, (1,)] = [0.0]
if x > 5.0 {
    let z = a
} elif x > 0.5 {
    let z = b
} else {
    let z = c
}
z

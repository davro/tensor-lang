// ============================================================================
// TensorLang: scalar literals in elif conditions (else branch taken)
// ============================================================================
// Same program as control_flow_elif_scalar_literal, but x=0.1 makes both the
// `if` (x > 5.0) and the `elif` (x > 0.5) false -> z = c. Guards against the
// literal fill producing a truthy value for every comparison.
// Expected value derived by hand.
//
// @EXPECTED
// {"z": [30.0]}
let x: Tensor[f32, (1,)] = [0.1]
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

// ============================================================================
// TensorLang: an f16 value that underflows to zero is falsy
// ============================================================================
// 0.00000001 is below half of the smallest f16 subnormal (about 6e-8), so it
// rounds to 0.0 when stored as f16 and the condition is false: z = b.
// (The language has no exponent literals, so the value is written in decimal.)
// Expected value derived by hand.
//
// @EXPECTED
// {"z": [20.0]}
let tiny: Tensor[f16, (1,)] = [0.00000001]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let z: Tensor[f32, (1,)] = [0.0]
if tiny {
    let z = a
} else {
    let z = b
}
z

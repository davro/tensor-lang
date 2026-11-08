// ============================================================================
// TensorLang: literal conditions on a 2-D tensor
// ============================================================================
// x = [[1,2],[3,4]]
//   x < 4.5 -> all four elements true            -> r1 = a (10)
//   x > 3.5 -> [[0,0],[0,1]], not all nonzero    -> r2 = b (20)
// With a (1,) literal the second column would be compared against whatever sits past
// the literal's one-element buffer, which flips r1. Expected values derived by hand.
//
// @EXPECTED
// {"r1": [10.0], "r2": [20.0]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let r1: Tensor[f32, (1,)] = [0.0]
let r2: Tensor[f32, (1,)] = [0.0]
if x < 4.5 {
    let r1 = a
} else {
    let r1 = b
}
if x > 3.5 {
    let r2 = a
} else {
    let r2 = b
}
r1
r2

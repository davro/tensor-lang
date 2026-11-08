// ============================================================================
// TensorLang: a multi-element condition is true only if EVERY element is nonzero
// ============================================================================
// both = [1, 2]  -> all nonzero -> true  -> r1 = a
// some = [1, 0]  -> one zero    -> false -> r2 = b
// Expected values derived by hand.
//
// @EXPECTED
// {"r1": [10.0], "r2": [20.0]}
let both: Tensor[f32, (2,)] = [1.0, 2.0]
let some: Tensor[f32, (2,)] = [1.0, 0.0]
let a: Tensor[f32, (1,)] = [10.0]
let b: Tensor[f32, (1,)] = [20.0]
let r1: Tensor[f32, (1,)] = [0.0]
let r2: Tensor[f32, (1,)] = [0.0]
if both {
    let r1 = a
} else {
    let r1 = b
}
if some {
    let r2 = a
} else {
    let r2 = b
}
r1
r2

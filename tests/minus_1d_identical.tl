// ============================================================================
// TensorLang: Minus 1D Identical
// ============================================================================
// Description: Tests minimum reduction
//
// Architecture Pattern:
//   'c' output
//
// Expected Output Variables:
//   c
//
// @EXPECTED
// {
//   "c": [
//     9.0,
//     18.0,
//     27.0,
//     36.0
//   ]
// }

let a: Tensor[f32, (4,)] = [10.0, 20.0, 30.0, 40.0]
let b: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0]
let c = minus(a, b)
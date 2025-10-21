// ============================================================================
// TensorLang: Minus 1D Scalar
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
//     -9,
//     -8,
//     -7,
//     -6,
//     -5
//   ]
// }

let a: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let b: Tensor[f32, (1,)] = [10.0]
let c = minus(a, b)
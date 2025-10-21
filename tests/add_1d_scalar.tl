// ============================================================================
// TensorLang: Add 1D Scalar
// ============================================================================
// Description: Tests addition operations
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
//     11.0,
//     12.0,
//     13.0,
//     14.0,
//     15.0
//   ]
// }

let a: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let b: Tensor[f32, (1,)] = [10.0]
let c = add(a, b)
// ============================================================================
// TensorLang: Add With Fill
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
//     [
//       5.0,
//       5.0
//     ],
//     [
//       5.0,
//       5.0
//     ]
//   ]
// }

let a : Tensor[f32, (2, 2)] = fill(2.0, (2, 2))
let b : Tensor[f32, (2, 2)] = fill(3.0, (2, 2))
let c = add(a, b)
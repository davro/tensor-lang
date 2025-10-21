// ============================================================================
// TensorLang: Fill
// ============================================================================
// Description: Tests tensor fill operation
//
// Architecture Pattern:
//   'x' output
//
// Expected Output Variables:
//   x
//
// @EXPECTED
// {
//   "x": [
//     [
//       1.0,
//       1.0,
//       1.0
//     ],
//     [
//       1.0,
//       1.0,
//       1.0
//     ]
//   ]
// }

let x : Tensor[f32, (2, 3)] = fill(1.0, (2, 3))
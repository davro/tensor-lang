// ============================================================================
// TensorLang: Minus 2D 1D Cols
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
//     [
//       9.0,
//       18.0,
//       27.0
//     ],
//     [
//       39.0,
//       48.0,
//       57.0
//     ]
//   ]
// }

let a: Tensor[f32, (2, 3)] = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
let b: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let c = minus(a, b)
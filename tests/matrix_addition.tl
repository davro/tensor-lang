// ============================================================================
// TensorLang: Matrix Addition
// ============================================================================
// Description: Tests addition operations
//
// Architecture Pattern:
//   'z' output
//
// Expected Output Variables:
//   z
//
// @EXPECTED
// {
//   "z": [
//     [
//       6.0,
//       8.0
//     ],
//     [
//       10.0,
//       12.0
//     ]
//   ]
// }

let x : Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let y : Tensor[f32, (2, 2)] = [[5.0, 6.0], [7.0, 8.0]]
let z = add(x, y)
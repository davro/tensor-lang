// ============================================================================
// TensorLang: Matrix Multiplication
// ============================================================================
// Description: Tests tensor operations
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
//       4.0,
//       5.0
//     ],
//     [
//       10.0,
//       11.0
//     ]
//   ]
// }

let x : Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let y : Tensor[f32, (3, 2)] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let z = matmul(x, y)
// ============================================================================
// TensorLang: Add 1D Identical
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
//     5.0,
//     7.0,
//     9.0
//   ]
// }

let x : Tensor[f32, (3)] = [1.0, 2.0, 3.0]
let y : Tensor[f32, (3)] = [4.0, 5.0, 6.0]
let z = add(x, y)
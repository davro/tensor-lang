// ============================================================================
// TensorLang: Multiply
// ============================================================================
// Description: Tests multiplication operations
//
// Architecture Pattern:
//   'mul' output
//
// Expected Output Variables:
//   mul
//
// @EXPECTED
// {
//   "mul": [
//     12.0,
//     32.0,
//     50.0
//   ]
// }

let x : Tensor[f32, (3)] = [6.0, 8.0, 10.0]
let y : Tensor[f32, (3)] = [2.0, 4.0, 5.0]
let mul = mult(x, y)   // [12.0, 32.0, 50.0]

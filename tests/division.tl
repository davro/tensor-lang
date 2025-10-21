// ============================================================================
// TensorLang: Division
// ============================================================================
// Description: Tests division operations
//
// Architecture Pattern:
//   'div' output
//
// Expected Output Variables:
//   div
//
// @EXPECTED
// {
//   "div": [
//     3.0,
//     2.0,
//     2.0
//   ]
// }

let x : Tensor[f32, (3)] = [6.0, 8.0, 10.0]
let y : Tensor[f32, (3)] = [2.0, 4.0, 5.0]
let div = div(x, y)    // [3.0, 2.0, 2.0]

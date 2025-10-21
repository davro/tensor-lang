// ============================================================================
// TensorLang: Relu
// ============================================================================
// Description: Tests ReLU activation
//
// Architecture Pattern:
//   'y' output
//
// Expected Output Variables:
//   y
//
// @EXPECTED
// {
//   "y": [
//     0.0,
//     2.0,
//     0.0,
//     4.0
//   ]
// }

let x : Tensor[f32, (4)] = [-1.0, 2.0, -3.0, 4.0]
let y = relu(x)
// ============================================================================
// TensorLang: Autograd basic
// ============================================================================
// Description: Tests Autograd basic
//
// Architecture Pattern:
//   'loss' output
//
// Expected Output Variables:
//   loss
//
// @EXPECTED
// {
//   "loss": 10.0,
//   "w": [[0.5, 0.5], [0.5, 0.5]]
// }

// Simple gradient test
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

let y = matmul(x, w)
let loss = sum(y)

backward(loss)
// ============================================================================
// TensorLang: Autograd Chain
// ============================================================================
// Description: Tests gradient flow through multiple operations
//
// Expected:
// - w1.grad and w2.grad both computed correctly
// - Chain rule applied correctly
//
// @EXPECTED
// {
//   "w1.grad": [[4.0, 4.0], [6.0, 6.0]],
//   "w2.grad": [[4.0, 4.0], [6.0, 6.0]]
// }

let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w1: Tensor[f32, (2, 2)] = [[1.0, 0.0], [0.0, 1.0]] with grad
let w2: Tensor[f32, (2, 2)] = [[0.5, 0.5], [0.5, 0.5]] with grad

let h = matmul(x, w1)
let y = matmul(h, w2)
let loss = sum(y)

backward(loss)


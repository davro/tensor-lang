// ============================================================================
// TensorLang: a parameter the loss never reaches gets a zero gradient
// ============================================================================
// w3 is declared `with grad` but is not part of loss1. Its gradient must be
// zeros (not an error, not stale data), so an sgd step leaves it unchanged,
// while w1 moves as usual:
//   w1 = 0.1 - 0.01 * [[4,4],[6,6]] = [[0.06, 0.06], [0.04, 0.04]]
//   w3 = 0.7 - 0.01 * 0           = [[0.7, 0.7], [0.7, 0.7]]
// Expected values derived by hand.
//
// @EXPECTED
// {"result1": [[0.06, 0.06], [0.04, 0.04]], "result3": [[0.7, 0.7], [0.7, 0.7]]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]

let w1: Tensor[f32, (2, 2)] = fill(0.1, (2, 2)) with grad
let w3: Tensor[f32, (2, 2)] = fill(0.7, (2, 2)) with grad

let y1 = matmul(x, w1)
let loss1 = sum(y1)
backward(loss1)

sgd(w1, w1_grad, 0.01)
sgd(w3, w3_grad, 0.01)

let result1 = w1
let result3 = w3

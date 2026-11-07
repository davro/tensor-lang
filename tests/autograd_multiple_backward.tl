// ============================================================================
// TensorLang: two sequential backward() passes with separate parameters
// ============================================================================
// x = [[1,2],[3,4]], w1 = 0.1, w2 = 0.2, lr = 0.01.
// For loss = sum(matmul(x, w)) the gradient is x^T @ ones(2,2) = [[4,4],[6,6]]
// for both parameters, so after one SGD step:
//   w1 = 0.1 - 0.01 * [[4,4],[6,6]] = [[0.06, 0.06], [0.04, 0.04]]
//   w2 = 0.2 - 0.01 * [[4,4],[6,6]] = [[0.16, 0.16], [0.14, 0.14]]
// A nonzero lr makes the test sensitive to wrong or missing gradients; the
// earlier lr=0.0 version could not detect them. Expected values derived by hand.
//
// @EXPECTED
// {"result1": [[0.06, 0.06], [0.04, 0.04]], "result2": [[0.16, 0.16], [0.14, 0.14]]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]

let w1: Tensor[f32, (2, 2)] = fill(0.1, (2, 2)) with grad
let w2: Tensor[f32, (2, 2)] = fill(0.2, (2, 2)) with grad

let y1 = matmul(x, w1)
let loss1 = sum(y1)
backward(loss1)

sgd(w1, w1_grad, 0.01)

let y2 = matmul(x, w2)
let loss2 = sum(y2)
backward(loss2)

sgd(w2, w2_grad, 0.01)

let result1 = w1
let result2 = w2

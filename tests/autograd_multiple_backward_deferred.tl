// ============================================================================
// TensorLang: two backward() passes, both SGD updates applied afterwards (EXPLORATORY)
// ============================================================================
// Same numbers as autograd_multiple_backward, but both backward() calls run
// before either parameter is updated. This checks that the first pass's
// gradient (w1_grad) survives the second pass. If it is lost or overwritten,
// w1 will not move to [[0.06, 0.06], [0.04, 0.04]]. Behaviour here is not yet
// specified in the README, so a failure is information, not necessarily a bug
// in the test. Expected values derived by hand.
//
// @EXPECTED
// {"result1": [[0.06, 0.06], [0.04, 0.04]], "result2": [[0.16, 0.16], [0.14, 0.14]]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]

let w1: Tensor[f32, (2, 2)] = fill(0.1, (2, 2)) with grad
let w2: Tensor[f32, (2, 2)] = fill(0.2, (2, 2)) with grad

let y1 = matmul(x, w1)
let loss1 = sum(y1)
backward(loss1)

let y2 = matmul(x, w2)
let loss2 = sum(y2)
backward(loss2)

sgd(w1, w1_grad, 0.01)
sgd(w2, w2_grad, 0.01)

let result1 = w1
let result2 = w2

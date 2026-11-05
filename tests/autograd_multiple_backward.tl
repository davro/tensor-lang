// @EXPECTED
// {"result": [[0.1, 0.1], [0.1, 0.1]]}
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]

let w1: Tensor[f32, (2, 2)] = fill(0.1, (2, 2)) with grad
let w2: Tensor[f32, (2, 2)] = fill(0.2, (2, 2)) with grad

let y1 = matmul(x, w1)
let loss1 = sum(y1)
backward(loss1)

sgd(w1, w1_grad, 0.0)

let y2 = matmul(x, w2)
let loss2 = sum(y2)
backward(loss2)

sgd(w2, w2_grad, 0.0)

let result = w1

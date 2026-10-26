let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let w: Tensor[f32, (2, 2)] = [[1.0, 1.0], [1.0, 1.0]] with grad

let y = matmul(x, w)
let loss = sum(y)

backward(loss)

// @EXPECTED
// {"loss": 20.0, "w_grad": [[4.0, 4.0], [6.0, 6.0]]}


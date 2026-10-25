// tests/layer_norm_autograd_basic.tl
let x: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]] with grad
let scale: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]

let y = layer_norm(x, axis=1, eps=1.0)
let z = mult(y, scale)
let loss = sum(z)

backward(loss)

// @EXPECTED
// {"loss": 0.8944270610809326,
//  "x_grad": [[-0.3577708, 0.35777092],
//             [-0.35777092, 0.35777092]]}


// tests/cross_entropy_autograd_probe.tl
let pred: Tensor[f32, (2, 2)] = [[0.7, 0.3], [0.2, 0.8]] with grad
let target: Tensor[f32, (2,)] = [0.0, 1.0]

let loss = cross_entropy(pred, target)
backward(loss)

// @EXPECTED
// {"pred_grad": [[-0.15, 0.15],
//                [0.1, -0.1]]}


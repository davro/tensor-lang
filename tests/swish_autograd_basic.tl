// tests/swish_autograd_basic.tl
let x: Tensor[f32, (4,)] = [-2.0, -1.0, 1.0, 2.0] with grad
let y = swish(x)
let loss = sum(y)
backward(loss)

// @EXPECTED
// {"loss": 1.9853053092956543,
//  "x_grad": [-0.09078424, 0.07232949, 0.92767054, 1.0907843]}


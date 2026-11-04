// @EXPECTED
// {"x_grad": [0.0, 0.0, 0.0, 0.0]}
let x: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0] with grad
let y = any(x)
let loss = y
backward(loss)
x_grad

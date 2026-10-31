// tests/mixed_precision_autograd.tl
let x: Tensor[f16, (4,)] = [-2.0, -1.0, 1.0, 2.0] with grad
let y = swish(x)
let loss = sum(y)
backward(loss)

// @EXPECTED
// {
//  "loss": 1.9853515625,
//  "x_grad": [-0.09075927734375, 0.072265625, 0.927734375, 1.0908203125]
// }
// tests/gelu_autograd_basic.tl
let x: Tensor[f32, (4,)] = [-2.0, -1.0, 1.0, 2.0] with grad
let y = gelu(x)
let loss = sum(y)
backward(loss)

// @EXPECTED
// {"loss": 2.591688871383667,
//  "x_grad": [-0.0852318, -0.08331548, 1.0833155, 1.0852318]}


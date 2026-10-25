// tests/broadcast_scalar_basic.tl
let x: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0]
let lr: Tensor[f32, (1, 1)] = [[0.1]]

let y_mul = mult(x, lr)
let y_sub_lr = minus(x, lr)
let y_sub_rl = minus(lr, x)
let y_div_lr = div(x, lr)
let y_div_rl = div(lr, x)

// @EXPECTED
// {"y_mul": [0.1, 0.2, 0.3, 0.4],
//  "y_sub_lr": [0.9, 1.9, 2.9, 3.9],
//  "y_sub_rl": [-0.9, -1.9, -2.9, -3.9],
//  "y_div_lr": [10.0, 20.0, 30.0, 40.0],
//  "y_div_rl": [0.1, 0.05, 0.03333333507180214, 0.025]}


# loss_scale: 2.0
// ============================================================================
// TensorLang: Loss Scaling — training loop
// ============================================================================
// Verifies loss scaling works inside a for-loop backward pass.
// 3 steps of gradient descent. grad = sum(x) = 10.0 each step.
// w: 1.0 -> 0.9 -> 0.8 -> 0.7
//
// @EXPECTED
// {
//   "w": [0.7]
// }

let x: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0]
let w: Tensor[f32, (1,)] = [1.0] with grad
let lr: Tensor[f32, (1,)] = [0.01]

for epoch in range(3) {
    let y = mult(x, w)
    let loss = sum(y)
    backward(loss)
    let step = mult(lr, w_grad)
    let w_new = minus(w, step)
    w = w_new
}


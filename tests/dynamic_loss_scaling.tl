# loss_scale: dynamic 1e38 1
// ============================================================================
// TensorLang: Dynamic Loss Scaling
// ============================================================================
// Uses an extremely large initial scale (1e38) that will overflow on the
// first backward pass. Dynamic scaling should detect the overflow, halve
// the scale, and skip the optimizer step (weights unchanged).
//
// @EXPECTED
// {
//   "w": [1.0]
// }

let x: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0]
let w: Tensor[f32, (1,)] = [1.0] with grad
let lr: Tensor[f32, (1,)] = [0.01]

for epoch in range(1) {
    let y = mult(x, w)
    let loss = sum(y)
    backward(loss)
    let step = mult(lr, w_grad)
    let w_new = minus(w, step)
    w = w_new
}


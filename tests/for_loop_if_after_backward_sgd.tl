// ============================================================================
// TensorLang: an `if` after backward() in a loop runs, and its sgd step sees the fresh gradient
// ============================================================================
// Model y = 2x, loss = mean((x*w - y)^2) = 7.5 * (w - 2)^2, grad = 15 * (w - 2). Start w = 0.
// `if loss > 1.0` takes the large step (lr 0.05) while the loss is above 1, the small one (lr 0.01) below it:
//   iteration 0: loss 30 -> w = 1.5      iteration 1: loss 1.875 -> w = 1.875   (both take the `if`)
//   iterations 2-5: loss 0.117 ... -> w = 1.89375, 1.9096875, 1.923234375, 1.93474921875   (the `else`)
// Always taking the `if` would give 1.9995; never taking it, 1.2457. The `if` comes AFTER backward().
// Recomputed independently in NumPy.
//
// @EXPECTED
// {"w1": [[1.93474921875]]}
let x:  Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y1: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w1: Tensor[f32, (1, 1)] = [[0.0]] with grad

for epoch in range(6) {
    let p1    = matmul(x, w1)
    let loss1 = mse_loss(p1, y1)
    backward(loss1)
    if loss1 > 1.0 {
        sgd(w1, w1_grad, 0.05)
    } else {
        sgd(w1, w1_grad, 0.01)
    }
}

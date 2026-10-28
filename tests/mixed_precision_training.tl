// tests/mixed_precision_training.tl
let x: Tensor[f16, (4, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
let w: Tensor[f16, (2, 1)] = [[0.5], [0.5]] with grad
let y_true: Tensor[f16, (4, 1)] = [[3.0], [7.0], [11.0], [15.0]]
let lr: Tensor[f16, (1, 1)] = [[0.01]]

for epoch in range(5) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
}
save(w, "w.npy")

// @EXPECTED
// {
//  "w": [[0.95166015625], [1.0400390625]]
// }
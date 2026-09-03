// Minimal practical example: train a tiny network to fit y = 2x
// Demonstrates: tensors, matmul, mse_loss, autograd, training loop

let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f32, (1, 1)] = [[0.5]] with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(5) {
    let y_pred    = matmul(x, w)
    let loss      = mse_loss(y_pred, y_true)
    backward(loss)
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
}

// Post-loop evaluation (distinct names to avoid kernel symbol clash)
let y_pred_final = matmul(x, w)
let loss_final   = mse_loss(y_pred_final, y_true)

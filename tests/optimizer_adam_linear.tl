// tests/optimizer_adam_linear.tl
let x: Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w: Tensor[f32, (1, 1)] = [[0.0]] with grad

for epoch in range(50) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    adam(w, w_grad, 0.05)
}

let y_pred_final = matmul(x, w)
let loss_final = mse_loss(y_pred_final, y_true)

// @EXPECTED
// {"loss_final": 0.09913282096385956}


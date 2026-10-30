// ============================================================================
// TensorLang: FP16 Master Weights — Adam optimizer
// ============================================================================
// Verifies that FP16 params with Adam use FP32 master weights internally.
// 5 steps of gradient descent on linear regression (y = 2x).
//
// @EXPECTED
// {
//   "w": [[0.9956]],
//   "loss_grad": 1,
//   "loss_final": 7.56640625
// }

let x:      Tensor[f16, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f16, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f16, (1, 1)] = [[0.5]] with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(5) {
    let y_pred = matmul(x, w)
    let loss = mse_loss(y_pred, y_true)
    backward(loss)
    adam(w, w_grad, 0.1)
}

let y_pred_final = matmul(x, w)
let loss_final = mse_loss(y_pred_final, y_true)


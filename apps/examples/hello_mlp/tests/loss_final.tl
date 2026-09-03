// =============================================================================
// TensorLang: hello_mlp — Loss Convergence Check
// =============================================================================
// Description: Verifies the hello_mlp app's training loop converges to the
//              documented final loss when fitting y = 2x with 5 epochs of
//              full-batch gradient descent.
//
// Architecture Pattern:
//   Single-weight linear model, no bias
//
// Expected Output Variables:
//   loss_final
//
// @EXPECTED
// {
//   "loss_final": 0.0164794921875
// }

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

let y_pred_final = matmul(x, w)
let loss_final   = mse_loss(y_pred_final, y_true)

// =============================================================================
// TensorLang: linear_regression — Loss Convergence Check
// =============================================================================
// Description: Verifies the linear_regression app's training loop converges
//              to the documented final loss when fitting y = 2.5x - 1.0 with
//              80 epochs of full-batch gradient descent over weight + bias.
//
// Architecture Pattern:
//   Single-weight, single-bias linear model
//
// Expected Output Variables:
//   loss_final
//
// @EXPECTED
// {
//   "loss_final": 0.022800864651799202
// }

let x: Tensor[f32, (6, 1)] = [
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
]

let y_true: Tensor[f32, (6, 1)] = [
    [1.5],
    [4.0],
    [6.5],
    [9.0],
    [11.5],
    [14.0]
]

let w: Tensor[f32, (1, 1)] = [[0.0]] with grad
let b: Tensor[f32, (1, 1)] = [[0.0]] with grad

let lr: Tensor[f32, (1, 1)] = [[0.05]]

for epoch in range(80) {
    let xw     = matmul(x, w)
    let y_pred = add(xw, b)
    let loss   = mse_loss(y_pred, y_true)

    backward(loss)

    let w_updated = minus(w, mult(lr, w_grad))
    let b_updated = minus(b, mult(lr, b_grad))

    w = w_updated
    b = b_updated
}

let xw_final     = matmul(x, w)
let y_pred_final = add(xw_final, b)
let loss_final   = mse_loss(y_pred_final, y_true)

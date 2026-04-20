// ============================================================================
// TensorLang: for loop — 20 iterations, tight convergence check
// ============================================================================
// Description:
//   Runs 20 SGD steps on linear regression (y = 2x) and verifies that
//   the weight converges very close to the true value of 2.0.
//   Tests that nothing accumulates incorrectly (gradients, buffers, memory)
//   over many iterations.
//
// Architecture:
//   forward: y_pred = matmul(x, w)
//   loss:    mse_loss(y_pred, y_true)
//   update:  w = w - lr * w_grad
//
// After 20 steps w converges to ~1.9999986 (within 1.5e-6 of true value 2.0)
// Loss drops from 16.875 to ~1.53e-11 — essentially zero.
//
// Step-by-step (selected, numpy float32, lr=0.1):
//   iter  1: loss=16.8750000000   w=[[2.75]]
//   iter  2: loss=4.2187500000    w=[[1.625]]
//   iter  5: loss=0.0659179688    w=[[2.046875]]
//   iter 10: loss=0.0000643730    w=[[1.9985352]]
//   iter 20: loss=0.0000000001    w=[[1.9999986]]
//   loss_final = 1.5347723092418164e-11
//   w_final    = [[1.9999985694885254]]
//
// @EXPECTED
// {
//   "loss_final": 1.5347723092418164e-11,
//   "w":          [[1.9999985694885254]]
// }

let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f32, (1, 1)] = [[0.5]] with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(20) {
    let y_pred    = matmul(x, w)
    let loss      = mse_loss(y_pred, y_true)
    backward(loss)
    let grad_step = mult(lr, w_grad)
    let w_updated = minus(w, grad_step)
    w = w_updated
}

// Post-loop evaluation — distinct names required
let y_pred_final = matmul(x, w)
let loss_final   = mse_loss(y_pred_final, y_true)

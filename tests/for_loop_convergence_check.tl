// ============================================================================
// TensorLang: for loop — 10-step convergence verification
// ============================================================================
// Description:
//   Runs 10 SGD steps on linear regression (y = 2x) and verifies both:
//     1. w ends up closer to 2.0 than it started (0.5)
//     2. loss_final is significantly lower than the starting loss (16.875)
//
//   This is a mid-range test between for_loop_basic (5 steps) and
//   for_loop_more_iterations (20 steps). At 10 steps the oscillation
//   pattern around w=2.0 is clearly visible in the weight trace.
//
// Architecture:
//   forward: y_pred = matmul(x, w)
//   loss:    mse_loss(y_pred, y_true)
//   update:  w = w - lr * w_grad
//
// Step-by-step (numpy float32, lr=0.1):
//   iter  1: loss=16.8750000000   w=[[2.75]]
//   iter  2: loss=4.2187500000    w=[[1.625]]
//   iter  3: loss=1.0546875000    w=[[2.1875]]
//   iter  4: loss=0.2636718750    w=[[1.90625]]
//   iter  5: loss=0.0659179688    w=[[2.046875]]
//   iter  6: loss=0.0164794922    w=[[1.9765625]]
//   iter  7: loss=0.0041198730    w=[[2.0117188]]
//   iter  8: loss=0.0010299683    w=[[1.9941406]]
//   iter  9: loss=0.0002574921    w=[[2.0029297]]
//   iter 10: loss=0.0000643730    w=[[1.9985352]]
//   loss_final = 1.609325408935547e-05
//   w_final    = [[1.99853515625]]
//
// @EXPECTED
// {
//   "loss_final": 1.609325408935547e-05,
//   "w":          [[1.99853515625]]
// }

let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f32, (1, 1)] = [[0.5]] with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(10) {
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

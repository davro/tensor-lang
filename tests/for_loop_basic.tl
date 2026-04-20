// ============================================================================
// TensorLang: for loop — basic training test
// ============================================================================
// Description:
//   Runs 5 SGD steps of linear regression (y = 2x) and verifies that
//   loss and w reach the mathematically exact float32 values.
//
// Architecture:
//   forward: y_pred = matmul(x, w)
//   loss:    mse_loss(y_pred, y_true)
//   update:  w = w - lr * w_grad   (via rebind)
//
// Step-by-step verification (computed independently in numpy float32):
//   iter 1: loss=16.875000    w_after=[[2.75]]
//   iter 2: loss=4.218750     w_after=[[1.625]]
//   iter 3: loss=1.054688     w_after=[[2.1875]]
//   iter 4: loss=0.263672     w_after=[[1.90625]]
//   iter 5: loss=0.065918     w_after=[[2.046875]]
//   final loss (post-loop mse on w=[[2.046875]]): 0.0164794921875
//
// Note: Post-loop tensors use distinct names (loss_final, y_pred_final)
// to avoid CUDA kernel name collisions with identically named loop-body tensors.
//
// @EXPECTED
// {
//   "loss_final": 0.0164794921875,
//   "w":          [[2.046875]]
// }

let x:      Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let w:      Tensor[f32, (1, 1)] = [[0.5]] with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(5) {
    let y_pred    = matmul(x, w)
    let loss      = mse_loss(y_pred, y_true)
    backward(loss)
    let grad_step = mult(lr, w_grad)
    let w_updated = minus(w, grad_step)
    w = w_updated
}

// Recompute with distinct names to avoid duplicate CUDA kernel symbols.
// Variables declared inside a for loop body are not persisted to .npy cache.
let y_pred_final = matmul(x, w)
let loss_final   = mse_loss(y_pred_final, y_true)
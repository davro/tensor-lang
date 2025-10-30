// ============================================================================
// TensorLang: Weight Update Test
// ============================================================================
// Description: Tests that weights can be updated using gradients
//
// Expected:
// - loss_before > loss_after (loss should decrease)
// - w_updated should be closer to optimal value
//
// The Difference Between w_grad and w.grad
// They are identified two different ways to access gradients:
// w_grad - A synthesized variable name (not actually saved as a tensor)
// w.grad - The actual gradient attribute attached to tensor w (w.grad.npy)
// 
// @EXPECTED
// {
//   "loss_before": 16.875,
//   "loss_after": 4.21875,
//   "w": [[0.5]],
//   "w_updated": [[2.75]],
//   "w.grad": [[-22.5]]
// }

let x: Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y_true: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]  // y = 2*x
let w: Tensor[f32, (1, 1)] = [[0.5]] with grad

// ============================================================================
// Iteration 1: Compute initial loss
// ============================================================================
let y_pred = matmul(x, w)
let loss_before = mse_loss(y_pred, y_true)

backward(loss_before)

// ============================================================================
// Weight Update Step
// ============================================================================
// After backward(), w_grad is automatically available as a tensor
let learning_rate: Tensor[f32, (1, 1)] = [[0.1]]
let update = mult(learning_rate, w_grad)
let w_updated = minus(w, update)

// ============================================================================
// Iteration 2: Compute loss after update
// ============================================================================
let y_pred_new = matmul(x, w_updated)
let loss_after = mse_loss(y_pred_new, y_true)
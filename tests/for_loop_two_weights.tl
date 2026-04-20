// ============================================================================
// TensorLang: for loop — two weight matrices updated per iteration
// ============================================================================
// Description:
//   Trains a 2-layer linear network for 5 SGD steps and verifies that
//   both w1 and w2 are correctly updated each iteration via chain rule.
//   Uses only supported autograd ops: matmul + mse_loss.
//
// Architecture:
//   h      = matmul(x, w1)           shape (4,2)
//   y_pred = matmul(h, w2)           shape (4,1)
//   loss   = mse_loss(y_pred, y_true)
//
// This tests:
//   - Multiple with-grad tensors coexisting in one loop
//   - Both rebind pointer swaps applied atomically each iteration
//   - Chain rule through two matmuls (w1_grad and w2_grad both non-zero)
//   - General broadcast multiply: lr(1,1) * w1_grad(2,2) and lr(1,1) * w2_grad(2,1)
//     (exercises the fixed compute_broadcast_index single-emission guard)
//
// Step-by-step (numpy float32, lr=0.05):
//   iter 1: loss=0.1250000000
//   iter 2: loss=0.1113330051
//   iter 3: loss=0.0991531536
//   iter 4: loss=0.0882881880
//   iter 5: loss=0.0785900801
//   loss_final = 0.06993050873279572
//
// @EXPECTED
// {
//   "loss_final": 0.06993050873279572,
//   "w1": [[0.556401789188385, 0.055159930139780045], [-0.05659163370728493, 0.44465771317481995]],
//   "w2": [[1.0303817987442017], [0.9745076298713684]]
// }

let x:      Tensor[f32, (4, 2)] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
let y_true: Tensor[f32, (4, 1)] = [[1.0], [0.0], [1.0], [0.0]]
let w1:     Tensor[f32, (2, 2)] = [[0.5, 0.0], [0.0, 0.5]] with grad
let w2:     Tensor[f32, (2, 1)] = [[1.0], [1.0]]            with grad
let lr:     Tensor[f32, (1, 1)] = [[0.05]]

for epoch in range(5) {
    let h          = matmul(x, w1)
    let y_pred     = matmul(h, w2)
    let loss       = mse_loss(y_pred, y_true)
    backward(loss)
    let w1_step    = mult(lr, w1_grad)
    let w1_updated = minus(w1, w1_step)
    let w2_step    = mult(lr, w2_grad)
    let w2_updated = minus(w2, w2_step)
    w1 = w1_updated
    w2 = w2_updated
}

// Post-loop evaluation — distinct names required to avoid CUDA symbol collisions
let h_final      = matmul(x, w1)
let y_pred_final = matmul(h_final, w2)
let loss_final   = mse_loss(y_pred_final, y_true)
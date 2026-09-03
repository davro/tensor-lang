// =============================================================================
// Linear Regression / Curve Fit
// =============================================================================
// Fit y = w*x + b to synthetic data that exactly follows y = 2.5x - 1.0
// Demonstrates:
//   - Multiple trainable parameters (weight + bias)
//   - matmul / add / mse_loss
//   - Autograd + training loop with weight rebinding
// =============================================================================

// Exact relationship: y = 2.5 * x - 1.0
let x: Tensor[f32, (6, 1)] = [
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
]

let y_true: Tensor[f32, (6, 1)] = [
    [1.5],   // 2.5*1 - 1
    [4.0],   // 2.5*2 - 1
    [6.5],   // 2.5*3 - 1
    [9.0],   // 2.5*4 - 1
    [11.5],  // 2.5*5 - 1
    [14.0]   // 2.5*6 - 1
]

// Trainable parameters (start from zero)
let w: Tensor[f32, (1, 1)] = [[0.0]] with grad
let b: Tensor[f32, (1, 1)] = [[0.0]] with grad

let lr: Tensor[f32, (1, 1)] = [[0.05]]

// Training loop
for epoch in range(80) {
    // Forward: y_pred = x * w + b
    let xw     = matmul(x, w)
    let y_pred = add(xw, b)
    let loss   = mse_loss(y_pred, y_true)

    backward(loss)

    // SGD updates
    let w_updated = minus(w, mult(lr, w_grad))
    let b_updated = minus(b, mult(lr, b_grad))

    w = w_updated
    b = b_updated
}

// Final evaluation (distinct names to avoid kernel symbol clashes)
let xw_final     = matmul(x, w)
let y_pred_final = add(xw_final, b)
let loss_final   = mse_loss(y_pred_final, y_true)
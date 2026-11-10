// =============================================================================
// TensorLang: adaptive_lr — Branch Taken in Some Iterations and Not Others
// =============================================================================
// Description: Two linear models trained in one loop. `if loss1 > 1.0` picks the learning rate for
//              model 1: the `if` runs in iterations 0-1, the `else` in iterations 2-5. Model 2 has its
//              own loss, its own backward pass and its own optimiser step in every iteration.
//
// Hand derivation (loss = mean((x*w - y)^2) = 7.5 * (w - target)^2, grad = 15 * (w - target)):
//   model 1 (target 2): w1 = 0 -> 1.5 -> 1.875 (lr 0.05, loss 30 then 1.875, both > 1)
//                       -> 1.89375 -> 1.9096875 -> 1.923234375 -> 1.93474921875 (lr 0.01)
//   model 2 (target 3, lr 0.02): error shrinks by 0.7 per step: 3 - 2.5 * 0.7^6 = 2.7058775
//   Always taking the `if` would give w1 = 1.9995; always the `else`, w1 = 1.2457.
// Values recomputed independently in NumPy (float64).
//
// Expected Output Variables:
//   w1, w2, loss1_final, loss2_final
//
// @EXPECTED
// {"w1": [[1.93474921875]], "w2": [[2.7058775]], "loss1_final": 0.03193248340301515, "loss2_final": 0.6488103375468741}

let x:  Tensor[f32, (4, 1)] = [[1.0], [2.0], [3.0], [4.0]]
let y1: Tensor[f32, (4, 1)] = [[2.0], [4.0], [6.0], [8.0]]
let y2: Tensor[f32, (4, 1)] = [[3.0], [6.0], [9.0], [12.0]]
let w1: Tensor[f32, (1, 1)] = [[0.0]] with grad
let w2: Tensor[f32, (1, 1)] = [[0.5]] with grad

for epoch in range(6) {
    // Model 1 fits y1 = 2x. Large steps while its loss is above the threshold, small steps below it.
    let p1    = matmul(x, w1)
    let loss1 = mse_loss(p1, y1)
    backward(loss1)
    if loss1 > 1.0 {
        sgd(w1, w1_grad, 0.05)
    } else {
        sgd(w1, w1_grad, 0.01)
    }

    // Model 2 fits y2 = 3x with its own loss and its own optimiser step, every iteration.
    let p2    = matmul(x, w2)
    let loss2 = mse_loss(p2, y2)
    backward(loss2)
    sgd(w2, w2_grad, 0.02)
}

// Post-loop evaluation (distinct names to avoid kernel symbol clash)
let p1_final    = matmul(x, w1)
let loss1_final = mse_loss(p1_final, y1)
let p2_final    = matmul(x, w2)
let loss2_final = mse_loss(p2_final, y2)

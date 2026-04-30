// ============================================================================
// TensorLang: Autograd training loop through ReLU
// ============================================================================
// Description:
//   Trains a 2-layer network with ReLU activation for 5 SGD steps.
//   This is the integration test — proves that:
//     1. relu gradient flows correctly through backward() each iteration
//     2. w1 and w2 both update correctly (chain rule through relu)
//     3. loss decreases monotonically across all 5 steps
//
//   Inside the loop, w1_grad and w2_grad are the synthesised variable names
//   (usable in TensorLang expressions). The @EXPECTED block uses the final
//   post-loop tensor values, not gradients, so naming is straightforward.
//
// Architecture:
//   h_pre  = matmul(x, w1)
//   h      = relu(h_pre)
//   y_pred = matmul(h, w2)
//   loss   = mse_loss(y_pred, y_true)
//
// Step-by-step (numpy float32, lr=0.1):
//   iter 1: loss=0.18750000
//   iter 2: loss=0.15185320
//   iter 3: loss=0.11763889
//   iter 4: loss=0.09305185
//   iter 5: loss=0.07592805
//   loss_final = 0.06409979611635208
//
// @EXPECTED
// {
//   "loss_final": 0.06409979611635208,
//   "w1":         [[0.6128250956535339, 0.6376994252204895], [-0.4550597369670868, 0.509461522102356]],
//   "w2":         [[0.5749542117118835], [0.6456303000450134]]
// }

let x:      Tensor[f32, (3, 2)] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let y_true: Tensor[f32, (3, 1)] = [[1.0], [0.0], [1.0]]
let w1:     Tensor[f32, (2, 2)] = [[0.5, 0.5], [-0.5, 0.5]] with grad
let w2:     Tensor[f32, (2, 1)] = [[0.5], [0.5]]             with grad
let lr:     Tensor[f32, (1, 1)] = [[0.1]]

for epoch in range(5) {
    let h_pre      = matmul(x, w1)
    let h          = relu(h_pre)
    let y_pred     = matmul(h, w2)
    let loss       = mse_loss(y_pred, y_true)
    backward(loss)
    let w1_updated = minus(w1, mult(lr, w1_grad))
    let w2_updated = minus(w2, mult(lr, w2_grad))
    w1 = w1_updated
    w2 = w2_updated
}

// Post-loop evaluation — distinct names required to avoid CUDA symbol collisions
let h_pre_final  = matmul(x, w1)
let h_final      = relu(h_pre_final)
let y_pred_final = matmul(h_final, w2)
let loss_final   = mse_loss(y_pred_final, y_true)

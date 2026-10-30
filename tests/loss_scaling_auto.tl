# loss_scale: auto
// ============================================================================
// TensorLang: Loss Scaling — auto-detect
// ============================================================================
// Verifies "auto" detects FP16 tensors and selects 65536.0 scale.
//
// @EXPECTED
// {
//   "loss": 30.0,
//   "x_grad": [2.0, 4.0, 6.0, 8.0]
// }

let x: Tensor[f16, (4,)] = [1.0, 2.0, 3.0, 4.0] with grad
let y = mult(x, x)
let loss = sum(y)
backward(loss)


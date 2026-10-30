# loss_scale: 65536
// ============================================================================
// TensorLang: Loss Scaling — FP16 upcast path
// ============================================================================
// Verifies FP16 tensors with large scale upcast gradients to FP32 during
// backward, then unscale correctly.
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


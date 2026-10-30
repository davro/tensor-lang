# loss_scale: 2.0
// ============================================================================
// TensorLang: Loss Scaling verification
// ============================================================================
// Verifies that scaling the loss by 2.0 and unscaling gradients produces
// the correct gradient values.
//
// Architecture:
//   y = x * x  (element-wise)
//   loss = sum(y)
//   d(loss)/d(x) = 2*x  (expected gradient)
//
// With loss_scale=2.0:
//   - Backward seeds gradient with 2.0 instead of 1.0
//   - All gradients are 2x during backward
//   - Unscaling divides by 2.0 before optimizer step
//   - Final gradient should match unscaled: [2.0, 4.0, 6.0, 8.0]
//
// @EXPECTED
// {
//   "loss": 30.0,
//   "x_grad": [2.0, 4.0, 6.0, 8.0]
// }

let x: Tensor[f32, (4,)] = [1.0, 2.0, 3.0, 4.0] with grad
let y = mult(x, x)
let loss = sum(y)
backward(loss)


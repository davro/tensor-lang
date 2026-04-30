// ============================================================================
// TensorLang: Autograd through ReLU
// ============================================================================
// Description:
//   Tests that gradients flow correctly through a relu activation.
//   Architecture: x -> matmul(w1) -> relu -> matmul(w2) -> sum -> loss
//
//   If relu grad is NOT wired up: w1.grad = [[0,0],[0,0],[0,0]] (zero)
//   If relu grad IS wired up:     w1.grad has non-zero values (correct)
//
// Gradient naming convention:
//   w1.grad  — saved to cache as w1.grad.npy, used in @EXPECTED
//   w1_grad  — synthesised variable name usable in TensorLang expressions
//
// Forward:
//   h_pre = matmul(x, w1)          shape (2,2)
//   h     = relu(h_pre)            shape (2,2)
//   y     = matmul(h, w2)          shape (2,1)
//   loss  = sum(y)
//
// Gradients (verified independently in numpy float32):
//   w1.grad = [[2.5, -2.5], [3.5, -3.5], [4.5, -4.5]]
//   w2.grad = [[1.6], [2.2]]
//
// @EXPECTED
// {
//   "loss":    -0.30000007152557373,
//   "w1.grad": [[2.5, -2.5], [3.5, -3.5], [4.5, -4.5]],
//   "w2.grad": [[1.600000023841858], [2.200000286102295]]
// }

let x:  Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let w1: Tensor[f32, (3, 2)] = [[0.1, 0.2], [-0.1, 0.3], [0.2, -0.1]] with grad
let w2: Tensor[f32, (2, 1)] = [[0.5], [-0.5]]                         with grad

let h_pre = matmul(x, w1)
let h     = relu(h_pre)
let y     = matmul(h, w2)
let loss  = sum(y)

backward(loss)
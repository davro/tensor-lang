// ============================================================================
// TensorLang: Autograd through Sigmoid
// ============================================================================
// Description:
//   Tests that gradients flow correctly through a sigmoid activation.
//   Architecture: x -> matmul(w) -> sigmoid -> sum -> loss
//
//   Sigmoid gradient: dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))
//   Implemented in autograd.py _backward_sigmoid using saved forward output.
//
// Gradient naming convention:
//   w.grad  — saved to cache as w.grad.npy, used in @EXPECTED
//   w_grad  — synthesised variable name usable in TensorLang expressions
//
// Gradients (verified independently in numpy float32):
//   w.grad = [[0.358, 0.369], [-0.358, -0.369]]
//
// @EXPECTED
// {
//   "loss":   2.2965080738067627,
//   "w.grad": [[0.3580707311630249, 0.36875781416893005], [-0.3580707311630249, -0.36875781416893005]]
// }

let x: Tensor[f32, (2, 2)] = [[1.0, -1.0], [0.5, -0.5]]
let w: Tensor[f32, (2, 2)] = [[0.3, 0.4], [-0.2, 0.1]] with grad

let h_pre = matmul(x, w)
let h     = sigmoid(h_pre)
let loss  = sum(h)

backward(loss)

// ============================================================================
// TensorLang: Autograd through Tanh
// ============================================================================
// Description:
//   Tests that gradients flow correctly through a tanh activation.
//   Architecture: x -> matmul(w) -> tanh -> sum -> loss
//
//   Tanh gradient: dL/dx = dL/dy * (1 - tanh(x)^2)
//   Implemented in autograd.py _backward_tanh using saved forward output.
//
// Gradient naming convention:
//   w.grad  — saved to cache as w.grad.npy, used in @EXPECTED
//   w_grad  — synthesised variable name usable in TensorLang expressions
//
// Gradients (verified independently in numpy float32):
//   w.grad = [[0.462, 0.441], [-0.359, -0.527]]
//
// @EXPECTED
// {
//   "loss":   -0.2771284878253937,
//   "w.grad": [[0.4624148905277252, 0.44126275181770325], [-0.3586283326148987, -0.5274967551231384]]
// }

let x: Tensor[f32, (2, 2)] = [[1.0, 0.5], [-0.5, -1.0]]
let w: Tensor[f32, (2, 2)] = [[0.2, -0.3], [0.4, 0.1]] with grad

let h_pre = matmul(x, w)
let h     = tanh(h_pre)
let loss  = sum(h)

backward(loss)

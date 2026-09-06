// ============================================================================
// TensorLang: Reshape a gradient tensor
// ============================================================================
// Description:
//   Regression test for a compiler gotcha: reshape() (and the other
//   single-tensor-field ops: transpose, sum/mean, softmax, slice,
//   layer_norm, batch_norm, instance_norm, max/min/argmax/argmin,
//   concat) looked tensors up with a plain `tensor_name not in env`
//   check during type checking, with no fallback for gradient tensors
//   (NAME_grad) — which only exist after backward() runs. So
//   `reshape(w_grad, ...)` was rejected as "Undefined tensor w_grad"
//   even though w_grad is a perfectly valid, real tensor by the time
//   the program actually runs.
//
//   _resolve_tensor_type() now falls back to inferring a NAME_grad
//   tensor's type from its base tensor NAME when it isn't already in
//   the type-checker's env, matching the fallback the add/matmul/etc.
//   arg-list checks already had.
//
// @EXPECTED
// {
//   "loss": 36.0,
//   "w_grad_flat": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
// }

let w: Tensor[f32, (2, 4)] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] with grad
let loss = sum(w)

backward(loss)

let w_grad_flat = reshape(w_grad, (8,))

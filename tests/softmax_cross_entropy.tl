// ============================================================================
// TensorLang: softmax_cross_entropy on logits
// ============================================================================
// Description: softmax_cross_entropy(logits, targets): mean of logsumexp - logit[target]; gradient (softmax - onehot)/N
//
// Expected:
// - loss and logits_grad computed in float64 NumPy
// - same numbers as the failing softmax -> cross_entropy chain, which this op gets right
//
// @EXPECTED
// {
//   "loss": 0.383943694,
//   "logits_grad": [[0.115611949, -0.18573414, 0.070122192], [0.100593686, 0.030298236, -0.130891923]]
// }

let logits: Tensor[f32, (2, 3)] = [[1.0, 2.0, 0.5], [0.2, -1.0, 1.5]] with grad
let target: Tensor[f32, (2,)] = [1.0, 2.0]

let loss = softmax_cross_entropy(logits, target)
backward(loss)

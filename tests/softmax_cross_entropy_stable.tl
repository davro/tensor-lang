// ============================================================================
// TensorLang: softmax_cross_entropy is numerically stable
// ============================================================================
// Description: logits of magnitude 1000: softmax followed by -log would overflow or hit the probability clamp
//
// Expected:
// - each row's loss is 2000 exactly (log(1 + e^-1000 + e^-2000) is 0 in f32), the gradient is (p - onehot)/2
//
// @EXPECTED
// {
//   "loss": 2000.000000000,
//   "logits_grad": [[0.5, 0.0, -0.5], [-0.5, 0.0, 0.5]]
// }

let logits: Tensor[f32, (2, 3)] = [[1000.0, 0.0, -1000.0], [-1000.0, 0.0, 1000.0]] with grad
let target: Tensor[f32, (2,)] = [2.0, 0.0]

let loss = softmax_cross_entropy(logits, target)
backward(loss)

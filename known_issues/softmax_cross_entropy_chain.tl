// ============================================================================
// TensorLang: softmax followed by cross_entropy, gradient to the logits
// ============================================================================
// Description: logits -> softmax -> cross_entropy -> backward. The gradient of
//              the mean negative log likelihood with respect to the logits is
//              (softmax(logits) - one_hot(target)) / N, whatever the route.
//
// Expected:
// - loss is the mean of -log(p[target]) over the 2 rows
// - logits_grad = (p - one_hot) / 2, computed in float64 NumPy
//
// @EXPECTED
// {
//   "loss": 0.383944,
//   "logits_grad": [[0.115612, -0.185734, 0.070122], [0.100594, 0.030298, -0.130892]]
// }

let logits: Tensor[f32, (2, 3)] = [[1.0, 2.0, 0.5], [0.2, -1.0, 1.5]] with grad
let target: Tensor[f32, (2,)] = [1.0, 2.0]

let probs = softmax(logits)
let loss = cross_entropy(probs, target)
backward(loss)

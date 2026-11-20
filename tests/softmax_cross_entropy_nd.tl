// ============================================================================
// TensorLang: softmax_cross_entropy with N-D logits
// ============================================================================
// Description: logits (2, 2, 3) with targets (2, 2): the leading dimensions are flattened to 4 rows
//
// Expected:
// - the mean is over all 4 rows, the gradient flows back through the reshape to the (4, 3) leaf
//
// @EXPECTED
// {
//   "loss": 0.697007506,
//   "flat_grad": [[0.057805974, -0.09286707, 0.035061096], [0.050296843, 0.015149118, -0.065445961], [-0.210529658, 0.079483509, 0.131046149], [-0.039974354, 0.031413246, 0.008561108]]
// }

let flat: Tensor[f32, (4, 3)] = [[1.0, 2.0, 0.5], [0.2, -1.0, 1.5], [-0.3, 0.4, 0.9], [2.0, 0.1, -1.2]] with grad
let logits = reshape(flat, (2, 2, 3))
let target: Tensor[f32, (2, 2)] = [[1.0, 2.0], [0.0, 0.0]]

let loss = softmax_cross_entropy(logits, target)
backward(loss)

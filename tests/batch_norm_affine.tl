// ============================================================================
// TensorLang: batch_norm with gamma and beta: y = gamma * (x - mean) / sqrt(var + eps) + beta
// ============================================================================
// Same input and statistics as batch_norm.tl, whose result is [[-1, -1, -1], [1, 1, 1]] (to 2e-6).
// gamma = [1, 2, 0.5], beta = [0, -1, 3]  ->  [[-1, -3, 2.5], [1, 1, 3.5]].
// Recomputed independently in NumPy.
//
// @EXPECTED
// {"normalized": [[-0.99999778, -2.9999956, 2.5000011], [0.99999778, 0.99999556, 3.4999989]]}
let input: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let running_mean: Tensor[f32, (3,)] = [2.5, 3.5, 4.5]
let running_var: Tensor[f32, (3,)] = [2.25, 2.25, 2.25]
let gamma: Tensor[f32, (3,)] = [1.0, 2.0, 0.5]
let beta: Tensor[f32, (3,)] = [0.0, -1.0, 3.0]
let normalized = batch_norm(input, running_mean, running_var, gamma, beta)

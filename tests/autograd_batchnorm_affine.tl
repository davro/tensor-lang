// ============================================================================
// TensorLang: batch_norm gradients for x, gamma and beta
// ============================================================================
// loss = sum(w * y), y = gamma * xhat + beta, xhat = (x - mean) / sqrt(var + eps):
//   dx = w * gamma / sqrt(var + eps)     dgamma = sum over the batch of w * xhat     dbeta = sum over the batch of w
// No gradient flows to the running statistics (they are constants here).
// Values recomputed in NumPy and checked against finite differences.
//
// @EXPECTED
// {"bn_loss": 30.499977, "bn_in_grad": [[0.66666519, 2.6666607, 0.99999778], [2.6666607, 6.6666519, 1.9999956]], "bn_gamma_grad": [2.9999933, 2.9999933, 2.9999933], "bn_beta_grad": [5.0, 7.0, 9.0]}
let bn_rm: Tensor[f32, (3,)] = [2.5, 3.5, 4.5]
let bn_rv: Tensor[f32, (3,)] = [2.25, 2.25, 2.25]
let bn_gamma: Tensor[f32, (3,)] = [1.0, 2.0, 0.5] with grad
let bn_beta: Tensor[f32, (3,)] = [0.0, -1.0, 3.0] with grad
let bn_in: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let bn_w: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let bn_out = batch_norm(bn_in, bn_rm, bn_rv, bn_gamma, bn_beta)
let bn_weighted = mult(bn_out, bn_w)
let bn_loss = sum(bn_weighted)
backward(bn_loss)

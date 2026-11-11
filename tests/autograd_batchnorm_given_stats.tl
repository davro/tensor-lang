// ============================================================================
// TensorLang: batch_norm backward differentiates the forward it ran, with the GIVEN statistics
// ============================================================================
// y = (x - mean) / sqrt(var + eps) with mean/var constants, loss = sum(w * y):
//   dx = w / sqrt(var + eps) per channel, i.e. [[1/2, 2/1], [3/2, 4/1]] (eps = 1e-5).
// The old backward recomputed the BATCH statistics and used the training-mode formula, which is different.
// Values recomputed in NumPy and checked against finite differences.
//
// @EXPECTED
// {"bn_loss": 17.999925, "bn_in_grad": [[0.49999938, 1.99999], [1.4999981, 3.99998]]}
let bn_rm: Tensor[f32, (2,)] = [0.5, 1.0]
let bn_rv: Tensor[f32, (2,)] = [4.0, 1.0]
let bn_in: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]] with grad
let bn_w: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let bn_out = batch_norm(bn_in, bn_rm, bn_rv)
let bn_weighted = mult(bn_out, bn_w)
let bn_loss = sum(bn_weighted)
backward(bn_loss)

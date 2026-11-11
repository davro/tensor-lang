// ============================================================================
// TensorLang: batch_norm gradients on an NCHW tensor: dgamma and dbeta reduce over N, H and W
// ============================================================================
// Shape (2, 3, 2, 2), channel at dim 1. loss = sum(w * y). dgamma[c] and dbeta[c] sum over every element of
// channel c; dx has the input's shape. Values recomputed in NumPy and checked against finite differences.
//
// @EXPECTED
// {"bn_loss": 203.65607, "x2_grad": [[0.14999925, 0.2999985, 0.44999775, 0.599997, 0.12499984, 0.14999981, 0.17499978, 0.19999975, 0.59999967, 0.6666663, 0.73333293, 0.79999956], [1.9499903, 2.0999895, 2.2499888, 2.399988, 0.42499947, 0.44999944, 0.47499941, 0.49999938, 1.3999992, 1.4666659, 1.5333325, 1.5999991]], "g_grad": [70.599647, 44.899944, 36.333313], "b_grad": [6.8, 10.0, 13.2]}
let x2: Tensor[f32, (2, 12)] = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]] with grad
let w2: Tensor[f32, (2, 12)] = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]]
let g: Tensor[f32, (3,)] = [1.5, 0.5, 2.0] with grad
let b: Tensor[f32, (3,)] = [0.1, -0.2, 0.3] with grad
let bn_rm: Tensor[f32, (3,)] = [2.5, 6.5, 10.5]
let bn_rv: Tensor[f32, (3,)] = [1.0, 4.0, 9.0]
let x4 = reshape(x2, (2, 3, 2, 2))
let w4 = reshape(w2, (2, 3, 2, 2))
let bn_out = batch_norm(x4, bn_rm, bn_rv, g, b)
let bn_weighted = mult(bn_out, w4)
let bn_loss = sum(bn_weighted)
backward(bn_loss)

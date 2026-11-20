// ============================================================================
// TensorLang: training a linear classifier with softmax_cross_entropy
// ============================================================================
// Description: matmul -> softmax_cross_entropy -> backward -> manual SGD step, three iterations; the loss must fall
//
// Expected:
// - loss is that of the third iteration (before its update); W is the final weight; W_grad the third gradient
//
// @EXPECTED
// {
//   "loss": 0.624479489,
//   "W_grad": [[0.002889469, -0.002889469], [0.212289438, -0.212289438], [-0.155674636, 0.155674636]],
//   "W": [[0.088757895, -0.188757896], [-0.046398394, 0.546398401], [0.16089052, 0.139109477]]
// }

let x: Tensor[f32, (4, 3)] = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
let target: Tensor[f32, (4,)] = [0.0, 1.0, 1.0, 0.0]
let W: Tensor[f32, (3, 2)] = [[0.1, -0.2], [0.3, 0.2], [-0.1, 0.4]] with grad
let lr: Tensor[f32, (1,)] = [0.5]

for step in range(3) {
    let logits = matmul(x, W)
    let loss = softmax_cross_entropy(logits, target)
    backward(loss)
    let W_updated = minus(W, mult(lr, W_grad))
    W = W_updated
}

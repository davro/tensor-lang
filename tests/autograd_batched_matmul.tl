// ============================================================================
// TensorLang: Batched Matmul Gradients
// ============================================================================
// Verifies gradients flow through batched (N-D) matmul.
// A: (2, 2, 3) filled with 1.0, B: (2, 3, 2) filled with 1.0
// C = A @ B -> (2, 2, 2), each element = 3.0
// loss = sum(C) = 2*2*2*3 = 24.0
// dL/dA = dL/dC @ B^T -> each element = 2*2 = 4.0 (2 batches * 2 cols)
// dL/dB = A^T @ dL/dC -> each element = 2*2 = 4.0 (2 batches * 2 rows)
//
// @EXPECTED
// {
//   "A_grad": [[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]],
//   "B_grad": [[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]
// }

let A: Tensor[f32, (2, 2, 3)] = fill(1.0, (2, 2, 3)) with grad
let B: Tensor[f32, (2, 3, 2)] = fill(1.0, (2, 3, 2)) with grad
let C = matmul(A, B)
let loss = sum(C)
backward(loss)


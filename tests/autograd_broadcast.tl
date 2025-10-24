// ============================================================================
// TensorLang: Autograd Broadcast
// ============================================================================
// Description: Tests gradient reduction through broadcasting
//
// Expected:
// - b.grad should sum over broadcast dimension (3 rows -> 1 row)
// - Verifies _unbroadcast() works correctly
//
// @EXPECTED
// {
//   "loss": 24.0,
//   "b.grad": [3.0, 3.0]
// }

let a: Tensor[f32, (3, 2)] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
let b: Tensor[f32, (2,)] = [0.5, 0.5] with grad

let y = add(a, b)
let loss = sum(y)

backward(loss)
let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let y = reshape(x, (3, 2))
let z = sum(y)
backward(z)

// @EXPECTED
// {
//   "z": 21.0
// }

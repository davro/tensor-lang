let a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let b = transpose(a)
let c = sum(b)
backward(c)

// @EXPECTED
// {
//   "c": 21.0
// }

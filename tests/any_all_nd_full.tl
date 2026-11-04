// @EXPECTED
// {
//   "a": 1.0,
//   "b": 0.0
// }
let m: Tensor[f32, (2, 2)] = [[0.0, 0.0], [0.0, 1.0]]
let a = any(m)
let b = all(m)
a
b

// @EXPECTED
// {
//   "a": 1.0,
//   "b": 0.0
// }
let data: Tensor[f32, (3,)] = [-1.0, 0.0, 2.0]
let a = any(data)
let b = all(data)
a
b

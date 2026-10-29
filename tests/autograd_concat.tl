let p: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let q: Tensor[f32, (2, 3)] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]] with grad
let cat = concat(p, q, axis=0)
let cats = sum(cat)
backward(cats)

// @EXPECTED
// {
//   "cats": 78.0
// }

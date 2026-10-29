let u: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]] with grad
let v: Tensor[f32, (2, 2)] = [[2.0, 1.0], [4.0, 3.0]]
let gt = greater(u, v)
let gts = sum(gt)
backward(gts)

// @EXPECTED
// {
//   "gts": 2.0
// }

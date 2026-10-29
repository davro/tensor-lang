let n: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let mn = min(n)
let mns = sum(mn)
backward(mns)

// @EXPECTED
// {
//   "mns": 1.0
// }

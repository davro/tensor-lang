let m: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let mx = max(m)
let mxs = sum(mx)
backward(mxs)

// @EXPECTED
// {
//   "mxs": 6.0
// }

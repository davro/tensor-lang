let in_in: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]] with grad
let in_out = instance_norm(in_in)
let in_s = sum(in_out)
backward(in_s)

// @EXPECTED
// {
//   "in_s": 0.0
// }

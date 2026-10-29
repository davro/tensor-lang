let bn_rm: Tensor[f32, (2,)] = [0.0, 0.0]
let bn_rv: Tensor[f32, (2,)] = [1.0, 1.0]
let bn_in: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]] with grad
let bn_out = batch_norm(bn_in, bn_rm, bn_rv)
let bn_s = sum(bn_out)
backward(bn_s)

// @EXPECTED
// {
//   "bn_s": 10.0
// }

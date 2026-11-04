// @EXPECTED
// {
//   "r0": [1.0, 0.0],
//   "r1": [0.0, 0.0]
// }
let m: Tensor[f32, (2, 2)] = [[0.0, 1.0], [0.0, 0.0]]
let r0 = any(m, axis=1)
let r1 = all(m, axis=0)
r0
r1

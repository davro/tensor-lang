let am: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] with grad
let amx = argmax(am)
let ams = sum(amx)
backward(ams)

// @EXPECTED
// {
//   "ams": 5.0
// }

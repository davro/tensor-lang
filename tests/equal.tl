// Test equal comparison with floating point tolerance
// For inputs [1, 2, 3] and [1, 2.0001, 3]:
// 1 == 1 = 1 (true), 2 == 2.0001 = 0 (false, outside tolerance), 3 == 3 = 1 (true)

let a: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let b: Tensor[f32, (3,)] = [1.0, 2.0001, 3.0]
let result = equal(a, b)
result

// Test greater than comparison
// For inputs [1, 3, 5] and [2, 2, 4]:
// 1 > 2 = 0 (false), 3 > 2 = 1 (true), 5 > 4 = 1 (true)

let a: Tensor[f32, (3,)] = [1.0, 3.0, 5.0]
let b: Tensor[f32, (3,)] = [2.0, 2.0, 4.0]
let result = greater(a, b)
result

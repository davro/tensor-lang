// Test less than comparison
// For inputs [1, 3, 5] and [2, 2, 4]:
// 1 < 2 = 1 (true), 3 < 2 = 0 (false), 5 < 4 = 0 (false)

let a: Tensor[f32, (3,)] = [1.0, 3.0, 5.0]
let b: Tensor[f32, (3,)] = [2.0, 2.0, 4.0]
let result = less(a, b)
result

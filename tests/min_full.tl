// Test full tensor min
// Expected result: 1.0 (minimum of all elements)

let data: Tensor[f32, (5,)] = [3.0, 1.0, 4.0, 2.0, 5.0]
let result = min(data)
result

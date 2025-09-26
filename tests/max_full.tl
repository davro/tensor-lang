// Test full tensor max
// Expected result: 5.0 (maximum of all elements)

let data: Tensor[f32, (5,)] = [1.0, 5.0, 2.0, 4.0, 3.0]
let result = max(data)
result

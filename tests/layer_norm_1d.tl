// Test 1D layer normalization
// Input: [1, 2, 3, 4, 5] -> mean=3, variance=2
// Expected: normalized to mean=0, std=1

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let result = layer_norm(data)
result

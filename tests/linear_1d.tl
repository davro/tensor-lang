// Test 1D linear layer (single sample)
// Input: [1, 2] (2 features)
// Weight: [[0.5, 0.2], [0.3, 0.8]] (2x2 weight matrix)
// Bias: [0.1, 0.05] (2 output features)
// Expected: [1*0.5 + 2*0.3 + 0.1, 1*0.2 + 2*0.8 + 0.05] = [1.2, 1.85]

let input: Tensor[f32, (2,)] = [1.0, 2.0]
let weight: Tensor[f32, (2, 2)] = [[0.5, 0.2], [0.3, 0.8]]
let bias: Tensor[f32, (2,)] = [0.1, 0.05]
let result = linear(input, weight, bias)
result

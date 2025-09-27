// Test 2D linear layer (batch of samples)
// Batch input: [[1, 2], [3, 4]] (2 samples, 2 features each)
// Weight: [[0.5, 0.2], [0.3, 0.8]] (2x2 weight matrix)
// Bias: [0.1, 0.05] (2 output features)
// Sample 1: [1*0.5 + 2*0.3 + 0.1, 1*0.2 + 2*0.8 + 0.05] = [1.2, 1.85]
// Sample 2: [3*0.5 + 4*0.3 + 0.1, 3*0.2 + 4*0.8 + 0.05] = [2.8, 3.85]

let input: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let weight: Tensor[f32, (2, 2)] = [[0.5, 0.2], [0.3, 0.8]]
let bias: Tensor[f32, (2,)] = [0.1, 0.05]
let result = linear(input, weight, bias)
result

// Test 2D layer normalization (row-wise normalization)
// Each row gets normalized independently to mean=0, std=1
// Common in transformer models for sequence processing

let data: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let result = layer_norm(data, axis=1)
result

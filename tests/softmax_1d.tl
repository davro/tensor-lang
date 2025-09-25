// Test 1D softmax activation function
// softmax(x) = exp(x) / sum(exp(x))
// For inputs [1, 2, 3], exp values: [2.718, 7.389, 20.086]
// Sum = 30.193, so softmax ≈ [0.090, 0.245, 0.665]
// Note: Values sum to 1.0

let data: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let result = softmax(data)
result

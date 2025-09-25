// Test tanh activation function
// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// Expected results for inputs [-2, -1, 0, 1, 2]:
// tanh(-2) ≈ -0.964, tanh(-1) ≈ -0.762, tanh(0) = 0.0
// tanh(1) ≈ 0.762, tanh(2) ≈ 0.964

let data: Tensor[f32, (5,)] = [-2.0, -1.0, 0.0, 1.0, 2.0]
let result = tanh(data)
result

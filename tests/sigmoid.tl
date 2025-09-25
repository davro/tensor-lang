// Test sigmoid activation function
// sigmoid(x) = 1 / (1 + exp(-x))
// Expected results for inputs [-2, -1, 0, 1, 2]:
// sigmoid(-2) ≈ 0.119, sigmoid(-1) ≈ 0.269, sigmoid(0) = 0.5
// sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881

let data: Tensor[f32, (5,)] = [-2.0, -1.0, 0.0, 1.0, 2.0]
let result = sigmoid(data)
result

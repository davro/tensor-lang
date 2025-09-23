// Simple linear model: y = Wx + b
let X : Tensor[f32, (5, 2)] = [
  [1.0, 2.0], 
  [2.0, 3.0], 
  [3.0, 4.0], 
  [4.0, 5.0], 
  [5.0, 6.0]
]
let W : Tensor[f32, (2, 1)] = [[0.5], [0.3]]
let b : Tensor[f32, (1)] = [1.0]
let linear_output = matmul(X, W)
let predictions = add(linear_output, b)

// Test mean squared error loss for regression
// Predictions vs targets: compute mean of squared differences
// Expected: mean((pred - target)²)

let predictions: Tensor[f32, (4,)] = [1.0, 2.5, 3.2, 1.8]
let targets: Tensor[f32, (4,)] = [1.2, 2.0, 3.0, 2.0]

// Differences: [-0.2, 0.5, 0.2, -0.2]
// Squared: [0.04, 0.25, 0.04, 0.04]
// Mean: (0.04 + 0.25 + 0.04 + 0.04) / 4 = 0.0925

let loss = mse_loss(predictions, targets)
loss

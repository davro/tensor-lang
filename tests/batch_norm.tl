// Test batch normalization
// Input: (2, 3) - 2 samples, 3 features each
// Running stats computed across the batch dimension for each feature
// Each feature gets normalized using its running statistics

let input: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

// Running statistics for each feature (computed from training data)
let running_mean: Tensor[f32, (3,)] = [2.5, 3.5, 4.5]  // Mean per feature
let running_var: Tensor[f32, (3,)] = [2.25, 2.25, 2.25]  // Variance per feature

let normalized = batch_norm(input, running_mean, running_var)
normalized
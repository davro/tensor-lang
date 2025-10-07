// Test different normalization techniques on same data
// Demonstrates the differences between layer norm, batch norm, and instance norm
// Common in comparing normalization strategies for neural networks

let input: Tensor[f32, (2, 4)] = [[1.0, 4.0, 7.0, 10.0], [2.0, 5.0, 8.0, 11.0]]

// Layer normalization: normalize within each sample (across features)
let layer_normalized = layer_norm(input, axis=1)

// Instance normalization: same as layer norm for 2D case
let instance_normalized = instance_norm(input)

// Batch normalization: normalize across samples (for each feature)
let running_mean: Tensor[f32, (4,)] = [1.5, 4.5, 7.5, 10.5]  // Feature means
let running_var: Tensor[f32, (4,)] = [0.25, 0.25, 0.25, 0.25]  // Feature variances
let batch_normalized = batch_norm(input, running_mean, running_var)

// Apply to neural network layer to see practical difference
let weights: Tensor[f32, (4, 2)] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
let bias: Tensor[f32, (2,)] = [0.1, 0.05]

// Compare outputs after different normalizations
let layer_output = linear(layer_normalized, weights, bias)
let instance_output = linear(instance_normalized, weights, bias)
let batch_output = linear(batch_normalized, weights, bias)

// Return one for comparison (batch norm output)
batch_output
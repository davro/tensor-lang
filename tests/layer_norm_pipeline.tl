// Test layer norm in neural network pipeline
// Demonstrates stabilizing effect of layer normalization in deep networks
// Pattern: Linear -> LayerNorm -> Activation -> Linear

let input: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

// First layer
let w1: Tensor[f32, (3, 4)] = [[0.1, -0.2, 0.3, 0.4],
                               [0.2, 0.1, -0.3, 0.2],
                               [-0.1, 0.4, 0.2, -0.2]]
let b1: Tensor[f32, (4,)] = [0.1, 0.05, -0.1, 0.2]

let hidden     = linear(input, w1, b1)      // Linear transformation
let normalized = layer_norm(hidden, axis=1) // Normalize features
let activated  = relu(normalized)           // Apply activation
activated

// Test multi-layer perceptron (MLP) using linear layers
// Build a simple 2-layer neural network: input -> hidden -> output
// This demonstrates real neural network construction in TensorLang
// expected results "final": [0.63297474, 0.3989121]

let input: Tensor[f32, (3,)] = [1.0, 0.5, -0.5]

// First layer: 3 input features -> 4 hidden features
let w1: Tensor[f32, (3, 4)] = [[0.2, -0.3, 0.1, 0.4],
                               [0.5, 0.2, -0.1, 0.3],
                               [-0.2, 0.4, 0.6, -0.1]]
let b1: Tensor[f32, (4,)] = [0.1, -0.05, 0.2, 0.0]

// Second layer: 4 hidden features -> 2 output features
let w2: Tensor[f32, (4, 2)] = [[0.3, -0.2],
                               [-0.1, 0.4],
                               [0.2, 0.1],
                               [0.5, -0.3]]
let b2: Tensor[f32, (2,)] = [0.05, -0.1]

// Forward pass
let hidden    = linear(input, w1, b1)     // First linear layer
let activated = relu(hidden)              // ReLU activation
let output    = linear(activated, w2, b2) // Second linear layer
let final     = sigmoid(output)           // Sigmoid output activation
final
// Test deep network with multiple layer norm applications
// Demonstrates layer norm's stabilizing effect in deeper architectures
// Pattern: 
// Input -> Linear -> LayerNorm -> ReLU -> 
//          Linear -> LayerNorm -> Sigmoid -> Classification

let input: Tensor[f32, (3, 4)] = [[1.0, -0.5, 2.0, 0.5],
                                  [-1.0, 1.5, -0.5, 2.0],
                                  [0.5, -1.0, 1.0, -0.5]]

// First layer: 4 -> 6 features
let w1: Tensor[f32, (4, 6)] = [[0.2, -0.3, 0.1, 0.4, -0.2, 0.3],
                               [0.5, 0.2, -0.1, 0.3, 0.4, -0.2],
                               [-0.2, 0.4, 0.6, -0.1, 0.2, 0.5],
                               [0.3, -0.1, 0.2, 0.5, -0.3, 0.1]]
let b1: Tensor[f32, (6,)] = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1]

// Second layer: 6 -> 4 features
let w2: Tensor[f32, (6, 4)] = [[0.3, -0.2, 0.1, 0.4],
                               [-0.1, 0.4, 0.2, -0.3],
                               [0.2, 0.1, -0.3, 0.2],
                               [0.5, -0.3, 0.4, 0.1],
                               [-0.2, 0.3, -0.1, 0.5],
                               [0.1, 0.2, 0.3, -0.2]]
let b2: Tensor[f32, (4,)] = [0.05, -0.1, 0.2, -0.05]

// Third layer: 4 -> 3 classes
let w3: Tensor[f32, (4, 3)] = [[0.4, -0.3, 0.2],
                               [0.1, 0.5, -0.4],
                               [-0.2, 0.3, 0.6],
                               [0.3, -0.1, 0.2]]
let b3: Tensor[f32, (3,)] = [0.1, 0.0, -0.1]

// Forward pass with multiple layer norms
let hidden1 = linear(input, w1, b1)       // First linear layer
let norm1 = layer_norm(hidden1, axis=1)   // First layer norm
let act1 = relu(norm1)                    // First activation

let hidden2 = linear(act1, w2, b2)        // Second linear layer
let norm2 = layer_norm(hidden2, axis=1)   // Second layer norm
let act2 = sigmoid(norm2)                 // Second activation

let logits = linear(act2, w3, b3)         // Final classification layer
let probs = softmax(logits, axis=1)       // Convert to probabilities
let predictions = argmax(probs, axis=1)   // Get predicted classes

// Output the final predictions
predictions

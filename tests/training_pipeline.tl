// ============================================================================
// TensorLang: Training Pipeline
// ============================================================================
// Description: Tests operation pipeline
//
// Architecture Pattern:
//   Loss value
//
// Expected Output Variables:
//   loss
//
// @EXPECTED
// {
//   "loss": 1.0891168117523193
// }

let input: Tensor[f32, (2, 3)] = [[1.0, 0.5, -0.2], [-0.5, 1.2, 0.8]]
let targets: Tensor[f32, (2,)] = [1.0, 0.0]  // True class labels

// Network weights
let w1: Tensor[f32, (3, 4)] = [[0.2, -0.1, 0.3, 0.1],
                               [0.1, 0.4, -0.2, 0.3],
                               [-0.3, 0.2, 0.5, -0.1]]
let b1: Tensor[f32, (4,)] = [0.1, 0.0, -0.1, 0.2]

let w2: Tensor[f32, (4, 2)] = [[0.3, -0.2],
                               [-0.1, 0.4],
                               [0.2, -0.3],
                               [0.5, 0.1]]
let b2: Tensor[f32, (2,)] = [0.05, -0.1]

// Forward pass
let hidden = linear(input, w1, b1)           // First layer
let normalized = layer_norm(hidden, axis=1)  // Normalize for stability
let activated = relu(normalized)             // Activation
let logits = linear(activated, w2, b2)       // Output layer
let probs = softmax(logits, axis=1)          // Convert to probabilities

// Loss computation
let loss = cross_entropy(probs, targets)     // Classification loss

// Model evaluation
let predictions = argmax(probs, axis=1)      // Predicted classes

// Output final loss value
loss

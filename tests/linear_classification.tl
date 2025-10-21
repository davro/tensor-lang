// ============================================================================
// TensorLang: Linear Classification
// ============================================================================
// Description: Tests linear transformation layer
//
// Architecture Pattern:
//   Classification output
//
// Expected Output Variables:
//   predictions
//
// @EXPECTED
// {
//   "predictions": [
//     0.0,
//     1.0
//   ]
// }

let features: Tensor[f32, (2, 3)] = [[1.0, 0.5, -0.2],    // Sample 1
                                     [-0.5, 1.5, 0.8]]    // Sample 2

// Classification weights: 3 features -> 3 classes
let class_weights: Tensor[f32, (3, 3)] = [[0.2, -0.1, 0.3],
                                          [0.4, 0.2, -0.3],
                                          [-0.1, 0.5, 0.1]]
let class_bias: Tensor[f32, (3,)] = [0.1, -0.05, 0.2]

// Forward pass for classification
let logits      = linear(features, class_weights, class_bias) // Linear classifier
let probs       = softmax(logits, axis=1)                     // Convert to probabilities
let predictions = argmax(probs, axis=1)                       // Get predicted classes
predictions

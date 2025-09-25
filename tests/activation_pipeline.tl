// Test chaining multiple activation functions
// Start with raw data, apply tanh, then sigmoid (simulating neural network layers)
//
// The mathematical progression is:
// Input: [-1.0, 0.0, 1.0, 2.0]
// tanh: [-0.7615942, 0.0, 0.7615942, 0.9640276]
// sigmoid(tanh): [0.31830028, 0.5, 0.68169975, 0.7239275]
//

let raw: Tensor[f32, (4,)] = [-1.0, 0.0, 1.0, 2.0]
let hidden = tanh(raw)          // First activation layer
let output = sigmoid(hidden)    // Second activation layer
output

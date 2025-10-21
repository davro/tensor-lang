// ============================================================================
// TensorLang: Activation Pipeline
// ============================================================================
// Description: Tests various activation functions
//
// Architecture Pattern:
//   Network output
//
// Expected Output Variables:
//   output
//
// @EXPECTED
// {
//   "output": [
//     0.31830028,
//     0.5,
//     0.68169975,
//     0.7239275
//   ]
// }

let raw: Tensor[f32, (4,)] = [-1.0, 0.0, 1.0, 2.0]
let hidden = tanh(raw)          // First activation layer
let output = sigmoid(hidden)    // Second activation layer
output

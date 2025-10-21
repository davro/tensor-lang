// ============================================================================
// TensorLang: Tensor Transforms
// ============================================================================
// Description: Tests tensor operations
//
// Architecture Pattern:
//   'result' output
//
// Expected Output Variables:
//   result
//
// @EXPECTED
// {
//   "result": [
//     [
//       13.3,
//       11.150001,
//       8.9
//     ]
//   ]
// }

let batch1: Tensor[f32, (2, 4)] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
let batch2: Tensor[f32, (1, 4)] = [[9.0, 10.0, 11.0, 12.0]]

// Concatenate batches (common in data loading)
let combined = concat(batch1, batch2, axis=0)    // Shape: (3, 4)

// Reshape for different processing (e.g., flatten for linear layer)
let flattened = reshape(combined, (12, 1))       // Shape: (12, 1)

// Transpose for matrix operations (common in ML)
let transposed = transpose(flattened)            // Shape: (1, 12)

// Apply linear transformation to demonstrate integration
let weight: Tensor[f32, (12, 3)] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
                                    [0.1, -0.1, 0.2], [0.3, -0.2, 0.1], [0.2, 0.1, -0.3],
                                    [-0.1, 0.4, 0.2], [0.5, -0.3, 0.1], [0.2, 0.3, -0.4],
                                    [0.3, 0.2, 0.1], [-0.2, 0.4, 0.3], [0.1, -0.1, 0.2]]
let bias: Tensor[f32, (3,)] = [0.1, 0.05, -0.1]

let result = linear(transposed, weight, bias)    // Shape: (1, 3)
result

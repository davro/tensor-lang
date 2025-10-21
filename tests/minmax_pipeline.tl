// ============================================================================
// TensorLang: Minmax Pipeline
// ============================================================================
// Description: Tests maximum reduction
//
// Architecture Pattern:
//   'overall_max' output
//
// Expected Output Variables:
//   overall_max
//
// @EXPECTED
// {
//   "overall_max": 0.7100998759269714
// }

let logits: Tensor[f32, (3, 4)] = [[1.0, 3.0, 2.0, 1.5],
                                   [2.5, 1.0, 4.0, 2.0],
                                   [1.8, 2.2, 1.5, 3.5]]

let probs = softmax(logits, axis=1)     // Convert to probabilities
let max_probs = max(probs, axis=1)      // Find max probability in each row
let overall_max = max(max_probs)        // Find highest confidence overall
overall_max

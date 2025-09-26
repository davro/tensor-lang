// Test min/max operations in a pipeline - common ML pattern
// Apply softmax to logits, find max probability per row, then get overall max

let logits: Tensor[f32, (3, 4)] = [[1.0, 3.0, 2.0, 1.5],
                                   [2.5, 1.0, 4.0, 2.0],
                                   [1.8, 2.2, 1.5, 3.5]]

let probs = softmax(logits, axis=1)     // Convert to probabilities
let max_probs = max(probs, axis=1)      // Find max probability in each row
let overall_max = max(max_probs)        // Find highest confidence overall
overall_max

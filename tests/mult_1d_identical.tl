// ============================================================================
// TensorLang: Deep Layer Norm
// ============================================================================
// Description: Tests deep neural network architecture
//
// Architecture Pattern:
//   'act2' output → Raw classification scores → Classification output → Softmax probabilities
//
// Expected Output Variables:
//   predictions, probs, logits, act2
//

// @EXPECTED
// {
//   "z": [
//     4, 10, 18
//   ]
// } 

let x : Tensor[f32, (3)] = [1.0, 2.0, 3.0]
let y : Tensor[f32, (3)] = [4.0, 5.0, 6.0]
let z = mult(x, y)
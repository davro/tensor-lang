// ============================================================================
// TensorLang: Sigmoid
// ============================================================================
// Description: Tests sigmoid activation
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
//     0.11920292,
//     0.2689414,
//     0.5,
//     0.7310586,
//     0.8807971
//   ]
// }

let data: Tensor[f32, (5,)] = [-2.0, -1.0, 0.0, 1.0, 2.0]
let result = sigmoid(data)
result

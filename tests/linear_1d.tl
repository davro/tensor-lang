// ============================================================================
// TensorLang: Linear 1D
// ============================================================================
// Description: Tests linear transformation layer
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
//     1.2,
//     1.85
//   ]
// }

let input: Tensor[f32, (2,)] = [1.0, 2.0]
let weight: Tensor[f32, (2, 2)] = [[0.5, 0.2], [0.3, 0.8]]
let bias: Tensor[f32, (2,)] = [0.1, 0.05]
let result = linear(input, weight, bias)
result

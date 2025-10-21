// ============================================================================
// TensorLang: Tanh
// ============================================================================
// Description: Tests tanh activation
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
//     -0.9640276,
//     -0.7615942,
//     0.0,
//     0.7615942,
//     0.9640276
//   ]
// }

let data: Tensor[f32, (5,)] = [-2.0, -1.0, 0.0, 1.0, 2.0]
let result = tanh(data)
result

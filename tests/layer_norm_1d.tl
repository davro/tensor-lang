// ============================================================================
// TensorLang: Layer Norm 1D
// ============================================================================
// Description: Tests layer normalization
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
//     -1.41421,
//     -0.707105,
//     0.0,
//     0.707105,
//     1.41421
//   ]
// }

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let result = layer_norm(data)
result

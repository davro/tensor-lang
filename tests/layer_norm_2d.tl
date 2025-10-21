// ============================================================================
// TensorLang: Layer Norm 2D
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
//     [
//       -1.2247449,
//       0.0,
//       1.2247449
//     ],
//     [
//       -1.2247449,
//       0.0,
//       1.2247449
//     ]
//   ]
// }

let data: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let result = layer_norm(data, axis=1)
result

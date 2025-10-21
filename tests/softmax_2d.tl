// ============================================================================
// TensorLang: Softmax 2D
// ============================================================================
// Description: Tests maximum reduction
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
//       0.2689414,
//       0.7310586
//     ],
//     [
//       0.2689414,
//       0.7310586
//     ]
//   ]
// }

let data: Tensor[f32, (2, 2)] = [[1.0, 2.0], [3.0, 4.0]]
let result = softmax(data, axis=1)
result

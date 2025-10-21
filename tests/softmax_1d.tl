// ============================================================================
// TensorLang: Softmax 1D
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
//     0.09003057,
//     0.24472848,
//     0.66524094
//   ]
// }

let data: Tensor[f32, (3,)] = [1.0, 2.0, 3.0]
let result = softmax(data)
result

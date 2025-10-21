// ============================================================================
// TensorLang: Slice 1D
// ============================================================================
// Description: Tests tensor slicing
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
//     2.0,
//     3.0,
//     4.0
//   ]
// }

let data: Tensor[f32, (5,)] = [1.0, 2.0, 3.0, 4.0, 5.0]
let result = data[1:4]
result

// ============================================================================
// TensorLang: Slice Pipeline
// ============================================================================
// Description: Tests operation pipeline
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
//     10.0,
//     16.0
//   ]
// }

let data: Tensor[f32, (4, 3)] = [[1.0, -2.0, 3.0],
                                 [4.0, -5.0, 6.0],
                                 [7.0, -8.0, 9.0],
                                 [10.0, -11.0, 12.0]]

let batch_slice = data[1:3, :]      // Extract middle 2 rows
let activated = relu(batch_slice)    // Apply ReLU
let result = sum(activated, axis=1)  // Sum each row
result

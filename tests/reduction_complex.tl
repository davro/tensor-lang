// ============================================================================
// TensorLang: Reduction Complex
// ============================================================================
// Description: Tests reduction operations
//
// Architecture Pattern:
//   'row_sums' output
//
// Expected Output Variables:
//   row_sums
//
// @EXPECTED
// {
//   "row_sums": [
//     6.0,
//     12.0,
//     22.0
//   ]
// }

let data: Tensor[f32, (3, 4)] = [[-1.0, 2.0, -3.0, 4.0],
                                 [5.0, -6.0, 7.0, -8.0],
                                 [-9.0, 10.0, -11.0, 12.0]]

let positive = relu(data)               // Apply ReLU (zeros out negatives)
let row_sums = sum(positive, axis=1)    // Sum each row
row_sums

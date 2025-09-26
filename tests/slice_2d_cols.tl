// Test 2D tensor column slicing
// Extract last 2 columns from 3x3 matrix
// matrix[:, 1:3] should return columns 1 and 2

let matrix: Tensor[f32, (3, 3)] = [[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]]
let result = matrix[:, 1:3]
result

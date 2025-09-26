// Test 2D tensor row slicing
// Extract first 2 rows from 3x3 matrix
// matrix[0:2, :] should return first 2 rows

let matrix: Tensor[f32, (3, 3)] = [[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]]
let result = matrix[0:2, :]
result

// Test min along axis 0 (column-wise min)
// Matrix:
// [3, 1, 4]
// [2, 5, 1]
// Expected result: [2, 1, 1] (min of each column)

let matrix: Tensor[f32, (2, 3)] = [[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]
let result = min(matrix, axis=0)
result
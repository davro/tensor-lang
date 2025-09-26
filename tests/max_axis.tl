
// Test max along axis 1 (row-wise max)
// Matrix:
// [1, 4, 2]
// [5, 3, 6]
// Expected result: [4, 6] (max of each row)

let matrix: Tensor[f32, (2, 3)] = [[1.0, 4.0, 2.0], [5.0, 3.0, 6.0]]
let result = max(matrix, axis=1)
result

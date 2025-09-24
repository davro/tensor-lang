// Test mean along axis 1 (mean each row)
// Matrix:
// [1, 2, 3]
// [4, 5, 6]
// Expected result: [2.0, 5.0] (row means: (1+2+3)/3=2.0, (4+5+6)/3=5.0)

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let row_means = mean(matrix, axis=1)
row_means

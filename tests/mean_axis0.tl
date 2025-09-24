// Test mean along axis 0 (mean each column)
// Matrix:
// [1, 2, 3]
// [4, 5, 6]
// Expected result: [2.5, 3.5, 4.5] (column means: (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5)

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let col_means = mean(matrix, axis=0)
col_means

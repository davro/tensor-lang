// Test sum along axis 0 (sum each column)
// Matrix:
// [1, 2, 3]
// [4, 5, 6]
// Expected result: [5, 7, 9] (column sums: 1+4=5, 2+5=7, 3+6=9)

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let col_sums = sum(matrix, axis=0)
col_sums

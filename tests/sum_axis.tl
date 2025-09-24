// Test sum along axis 1 (sum each row)
// Matrix:
// [1, 2, 3]
// [4, 5, 6]
// Expected result: [6, 15] (row sums: 1+2+3=6, 4+5+6=15)

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let row_sums = sum(matrix, axis=1)
row_sums

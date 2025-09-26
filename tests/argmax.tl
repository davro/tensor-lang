// Test argmax along axis 1 (index of max in each row)
// Matrix:
// [1, 4, 2]  -> max is 4 at index 1
// [5, 3, 6]  -> max is 6 at index 2
// Expected result: [1, 2] (indices of max elements in each row)

let matrix: Tensor[f32, (2, 3)] = [[1.0, 4.0, 2.0], [5.0, 3.0, 6.0]]
let result = argmax(matrix, axis=1)
result

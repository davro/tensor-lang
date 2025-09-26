// Test argmin along axis 1 (index of min in each row)
// Matrix:
// [3, 1, 4]  -> min is 1 at index 1
// [2, 5, 1]  -> min is 1 at index 2
// Expected result: [1, 2] (indices of min elements in each row)

let matrix: Tensor[f32, (2, 3)] = [[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]
let result = argmin(matrix, axis=1)
result

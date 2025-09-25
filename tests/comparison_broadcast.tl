// Test broadcasting comparison operations
// Matrix [[1,2,3], [4,5,6]] > vector [2,2,2]
// Row 1: [1>2, 2>2, 3>2] = [0, 0, 1]
// Row 2: [4>2, 5>2, 6>2] = [1, 1, 1]

let matrix: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let threshold: Tensor[f32, (3,)] = [2.0, 2.0, 2.0]
let result = greater(matrix, threshold)
result

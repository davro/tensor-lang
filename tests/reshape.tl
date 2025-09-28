// Test tensor reshape operation
// Original: (2, 6) -> 12 elements
// Reshape to: (3, 4) -> 12 elements (same total)
// Elements: [1,2,3,4,5,6,7,8,9,10,11,12]

let matrix: Tensor[f32, (2, 6)] = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
let reshaped = reshape(matrix, (3, 4))
reshaped
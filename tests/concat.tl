// Test tensor concatenation along axis 0 (row-wise)
// Tensor A (2x3): [[1,2,3], [4,5,6]]
// Tensor B (1x3): [[7,8,9]]
// Concatenated (3x3): [[1,2,3], [4,5,6], [7,8,9]]

let tensor_a: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let tensor_b: Tensor[f32, (1, 3)] = [[7.0, 8.0, 9.0]]
let concatenated = concat(tensor_a, tensor_b, axis=0)
concatenated
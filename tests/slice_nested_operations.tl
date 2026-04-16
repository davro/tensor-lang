// ============================================================================
// TensorLang: Nested Slice Operations
// ============================================================================
// Description: Tests complex nested operations with slicing
//
// Test Cases:
//   1. Slice -> Activation -> Slice again
//   2. Multiple slices combined
//   3. Slice with reduction operations
//   4. Slice in matrix multiplication
//
// Expected Output Variables:
//   nested_slice, combined_slices, slice_sum, matmul_slice
//
// @EXPECTED
// {
//   "nested_slice": [3.0, 4.0],
//   "combined_slices": [6.0, 8.0],
//   "slice_sum": 9.0,
//   "matmul_slice": [[15.0, 22.0], [23.0, 34.0]]
// }

// Test data
let data : Tensor[f32, (8)] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
let matrix_a : Tensor[f32, (4, 2)] = [
    1.0, 2.0,
    3.0, 4.0,
    5.0, 6.0,
    7.0, 8.0
]
let matrix_b : Tensor[f32, (2, 2)] = [
    1.0, 2.0,
    3.0, 4.0
]

// Test Case 1: Slice -> Activation -> Slice again
let slice_step1 = data[1:6]  // [2.0, 3.0, 4.0, 5.0, 6.0]
let activated = relu(slice_step1)
let nested_slice = activated[1:3]  // [3.0, 4.0]

// Test Case 2: Multiple slices combined
let slice_a = data[0:2]  // [1.0, 2.0]
let slice_b = data[4:6]  // [5.0, 6.0]
let combined_slices = add(slice_a, slice_b)

// Test Case 3: Slice with reduction
let slice_for_sum = data[1:4]  // [2.0, 3.0, 4.0]
let slice_sum = sum(slice_for_sum)

// Test Case 4: Slice matrices before matmul
let matrix_a_slice = matrix_a[1:3, 0:2]  // [[3.0, 4.0], [5.0, 6.0]]
let matmul_slice = matmul(matrix_a_slice, matrix_b)
// ============================================================================
// TensorLang: Slice with Operations
// ============================================================================
// Description: Tests slicing combined with tensor operations
//
// Test Cases:
//   1. Slice then add
//   2. Slice then multiply
//   3. Slice then activation (ReLU)
//   4. Slice two tensors and operate
//   5. Chain operations on slices
//
// Expected Output Variables:
//   slice_add, slice_mult, slice_relu, slices_combined, chained
//
// @EXPECTED
// {
//   "slice_add": [7.0, 9.0, 11.0],
//   "slice_mult": [12.0, 20.0, 30.0],
//   "slice_relu": [0.0, 0.0, 1.0],
//   "slices_combined": [4.0, 6.0, 8.0],
//   "chained": [0.0, 5.0, 12.0]
// }

// Test data
let data1 : Tensor[f32, (6)] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
let data2 : Tensor[f32, (6)] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
let data3 : Tensor[f32, (5)] = [-1.0, 0.0, 1.0, 2.0, 3.0]

// Test Case 1: Slice then add scalar-like tensor
let slice1 = data1[1:4]  // [2.0, 3.0, 4.0]
let scalar_add : Tensor[f32, (3)] = [5.0, 6.0, 7.0]
let slice_add = add(slice1, scalar_add)

// Test Case 2: Slice then multiply
let slice2 = data1[2:5]  // [3.0, 4.0, 5.0]
let multiplier : Tensor[f32, (3)] = [4.0, 5.0, 6.0]
let slice_mult = mult(slice2, multiplier)

// Test Case 3: Slice then ReLU (remove negatives)
let slice3 = data3[0:3]  // [-1.0, 0.0, 1.0]
let slice_relu = relu(slice3)

// Test Case 4: Slice two tensors and add them
let slice4a = data1[0:3]  // [1.0, 2.0, 3.0]
let slice4b = data2[2:5]  // [30.0, 40.0, 50.0] divided by 10
let divisor : Tensor[f32, (3)] = [10.0, 10.0, 10.0]
let slice4b_scaled = div(slice4b, divisor)
let slices_combined = add(slice4a, slice4b_scaled)

// Test Case 5: Chain operations
let slice5 = data3[1:4]  // [0.0, 1.0, 2.0]
let doubled = mult(slice5, multiplier)
let chained = relu(doubled)
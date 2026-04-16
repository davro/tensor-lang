// ============================================================================
// TensorLang: Slice Edge Cases
// ============================================================================
// Description: Tests edge cases and boundary conditions for slicing
//
// Test Cases:
//   1. Single element slice
//   2. First element
//   3. Last element
//   4. Empty-like slice (single point)
//   5. Boundary slices (0:1, n-1:n)
//
// Expected Output Variables:
//   single_mid, first_elem, last_elem, boundary_start, boundary_end
//
// @EXPECTED
// {
//   "single_mid": [3.0],
//   "first_elem": [1.0],
//   "last_elem": [5.0],
//   "boundary_start": [1.0],
//   "boundary_end": [5.0]
// }

// Test data
let data : Tensor[f32, (5)] = [1.0, 2.0, 3.0, 4.0, 5.0]

// Test Case 1: Single element from middle
let single_mid = data[2:3]

// Test Case 2: First element
let first_elem = data[0:1]

// Test Case 3: Last element
let last_elem = data[4:5]

// Test Case 4: Boundary at start
let boundary_start = data[0:1]

// Test Case 5: Boundary at end
let boundary_end = data[4:5]

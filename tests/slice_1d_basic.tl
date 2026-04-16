// ============================================================================
// TensorLang: 1D Slice Basic Operations
// ============================================================================
// Description: Tests basic 1D slicing operations
//
// Test Cases:
//   1. Simple index access
//   2. Range slicing [start:end]
//   3. From start [:end]
//   4. To end [start:]
//   5. Full slice [:]
//
// Expected Output Variables:
//   single_elem, range_slice, from_start, to_end, full_slice
//
// @EXPECTED
// {
//   "single_elem": [2.0],
//   "range_slice": [2.0, 3.0, 4.0],
//   "from_start": [1.0, 2.0, 3.0],
//   "to_end": [4.0, 5.0, 6.0],
//   "full_slice": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
// }

// Define test data
let data : Tensor[f32, (6)] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

// Test Case 1: Single index access
let single_elem = data[1:2]

// Test Case 2: Range slice [start:end]
let range_slice = data[1:4]

// Test Case 3: From start [:end]
let from_start = data[0:3]

// Test Case 4: To end [start:]
let to_end = data[3:6]

// Test Case 5: Full slice [:]
let full_slice = data[0:6]
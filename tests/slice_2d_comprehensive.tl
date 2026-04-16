
// ============================================================================
// TensorLang: 2D Slice Comprehensive Test
// ============================================================================
// Description: Tests all 2D slicing patterns
//
// Test Cases:
//   1. Row selection
//   2. Column selection
//   3. Submatrix extraction
//   4. Single row, range of columns
//   5. Range of rows, single column
//   6. Full row slice with all columns
//   7. All rows with column range
//
// Expected Output Variables:
//   row_slice, col_slice, submatrix, row_partial_col,
//   partial_row_col, full_row, all_rows_partial_col
//
// @EXPECTED
// {
//   "row_slice": [[5.0, 6.0, 7.0, 8.0]],
//   "col_slice": [[2.0], [6.0], [10.0]],
//   "submatrix": [[6.0, 7.0], [10.0, 11.0]],
//   "row_partial_col": [[6.0, 7.0]],
//   "partial_row_col": [[6.0], [10.0]],
//   "full_row": [[5.0, 6.0, 7.0, 8.0]],
//   "all_rows_partial_col": [[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]
// }

// Define test matrix (3x4)
// [[ 1,  2,  3,  4],
//  [ 5,  6,  7,  8],
//  [ 9, 10, 11, 12]]
let matrix : Tensor[f32, (3, 4)] = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0
]

// Test Case 1: Extract single row (row 1)
let row_slice = matrix[1:2, 0:4]

// Test Case 2: Extract single column (column 1)
let col_slice = matrix[0:3, 1:2]

// Test Case 3: Extract submatrix (rows 1-2, cols 1-2)
let submatrix = matrix[1:3, 1:3]

// Test Case 4: Single row with partial columns
let row_partial_col = matrix[1:2, 1:3]

// Test Case 5: Partial rows with single column
let partial_row_col = matrix[1:3, 1:2]

// Test Case 6: Full row slice
let full_row = matrix[1:2, 0:4]

// Test Case 7: All rows with partial columns
let all_rows_partial_col = matrix[0:3, 1:3]

// ============================================================================
// TensorLang: Broadcasting a 1D tensor
// ============================================================================
// Description: Broadcasting a 1D tensor across columns
//
// Architecture Pattern:
//   'result' output
//
// Expected Output Variables:
//   result
//
// @EXPECTED
// {
//   "result": [
//     [ -9,  -8,  -7,  -6],
//     [-15, -14, -13, -12],
//     [-21, -20, -19, -18]
//   ]
// }

// tests/broadcast_minus_rows.tl
// Tests broadcasting a 1D tensor across columns
let data: Tensor[f32, (3, 4)] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]
let row_vals: Tensor[f32, (3,)] = [10.0, 20.0, 30.0]
let result = minus(data, row_vals)
result

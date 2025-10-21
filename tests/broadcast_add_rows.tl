// ============================================================================
// TensorLang: Broadcast Add Rows
// ============================================================================
// Description: Tests addition operations
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
//     [
//       11.0,
//       12.0,
//       13.0,
//       14.0
//     ],
//     [
//       25.0,
//       26.0,
//       27.0,
//       28.0
//     ],
//     [
//       39,
//       40,
//       41,
//       42
//     ]
//   ]
// }

let data: Tensor[f32, (3, 4)] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]
let row_vals: Tensor[f32, (3,)] = [10.0, 20.0, 30.0]
let result = add(data, row_vals)
result

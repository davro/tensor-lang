// ============================================================================
// TensorLang: Function Broadcast Add
// ============================================================================
// Description: Tests addition operations
//
// Architecture Pattern:
//   'adjusted' output
//
// Expected Output Variables:
//   adjusted
//
// @EXPECTED
// {
//   "adjusted": [
//     [
//       101,
//       102,
//       103,
//       104
//     ],
//     [
//       205,
//       206,
//       207,
//       208
//     ],
//     [
//       309,
//       310,
//       311,
//       312
//     ]
//   ]
// }

fn add_bias_rows(data: Tensor[M, N], bias: Tensor[M, 1]) -> Tensor[M, N] {
    return add(data, bias)
}

let features: Tensor[f32, (3, 4)] = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]

let row_bias: Tensor[f32, (3,)] = [100.0, 200.0, 300.0]

let adjusted = add_bias_rows(features, row_bias)
adjusted

// ============================================================================
// TensorLang: Function Simple
// ============================================================================
// Description: Tests custom function definition
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
//       1.5,
//       2.5,
//       3.5
//     ],
//     [
//       4.5,
//       5.5,
//       6.5
//     ]
//   ]
// }

fn add_tensors(a: Tensor[M, N], b: Tensor[M, N]) -> Tensor[M, N] {
    return add(a, b)
}

let x: Tensor[f32, (2, 3)] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let y: Tensor[f32, (2, 3)] = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
let result = add_tensors(x, y)
result